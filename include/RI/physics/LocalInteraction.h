#pragma once

#include "../global/Tensor.h"
#include "../global/Tensor_Multiply.h"

#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace RI
{
namespace Local_Interaction
{

/*!
 * Generic, occupation-independent local interaction integrals.
 *
 * A pair vertex is a row-major matrix Q(ab,mu), ab=a*n_orbitals+b.
 * The integral convention is bilinear:
 *   I(ab,cd) = integral phi_a* phi_b v phi_c* phi_d = Q(ab) V Q(cd)^T.
 * There is NO implicit complex conjugation of the left vertex. Kernel units,
 * Fourier signs/weights, screening, hybrid fractions and model averages belong
 * to the caller. The routines do not construct an occupied-density matrix.
 */

namespace detail
{
inline std::size_t product(std::size_t a, std::size_t b)
{
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::invalid_argument("LocalInteraction: tensor size overflows");
    return a * b;
}

inline std::size_t pair_count(std::size_t n)
{
    if (!n) throw std::invalid_argument("LocalInteraction: orbital count is zero");
    const std::size_t result = product(n, n);
    if (result > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("LocalInteraction: pair count exceeds BLAS integer range");
    return result;
}

template<typename T>
void validate(const Tensor<T>& tensor, std::size_t rank, const char* name)
{
    if (tensor.shape.size() != rank)
        throw std::invalid_argument(std::string("LocalInteraction: invalid rank for ") + name);
    std::size_t size = 1;
    for (std::size_t i = 0; i != rank; ++i)
    {
        if (!tensor.shape[i] ||
            tensor.shape[i] > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument(std::string("LocalInteraction: invalid dimension for ") + name);
        size = product(size, tensor.shape[i]);
    }
    if (!tensor.data || tensor.data->size() != size)
        throw std::invalid_argument(std::string("LocalInteraction: invalid storage for ") + name);
    for (std::size_t i = 0; i != size; ++i)
        if (!std::isfinite(std::real(tensor.ptr()[i])) ||
            !std::isfinite(std::imag(tensor.ptr()[i])))
            throw std::invalid_argument(std::string("LocalInteraction: non-finite ") + name);
}

template<typename Tout, typename Tin>
typename std::enable_if<std::is_same<Tout, Tin>::value, Tensor<Tout>>::type
promote(const Tensor<Tin>& input)
{
    return input; // Tensor copies share storage; the contraction never mutates it.
}

template<typename Tout, typename Tin>
typename std::enable_if<!std::is_same<Tout, Tin>::value, Tensor<Tout>>::type
promote(const Tensor<Tin>& input)
{
    Tensor<Tout> result(input.shape);
    for (std::size_t i = 0; i != input.data->size(); ++i)
        result.ptr()[i] = static_cast<Tout>(input.ptr()[i]);
    return result;
}

template<typename T>
T conjugate(const T& value) { return value; }

template<typename T>
std::complex<T> conjugate(const std::complex<T>& value) { return std::conj(value); }

template<typename T>
T checked_add(T a, T b)
{
    static_assert(std::is_integral<T>::value && std::is_signed<T>::value,
                  "LocalInteraction cell indices must be signed integers");
    if ((b > 0 && a > std::numeric_limits<T>::max() - b) ||
        (b < 0 && a < std::numeric_limits<T>::min() - b))
        throw std::invalid_argument("LocalInteraction: cell addition overflows");
    return a + b;
}

template<typename T, std::size_t N>
std::array<T, N> add(const std::array<T, N>& a, const std::array<T, N>& b)
{
    std::array<T, N> result;
    for (std::size_t d = 0; d != N; ++d) result[d] = checked_add(a[d], b[d]);
    return result;
}

template<typename T, std::size_t N>
std::array<T, N> subtract(const std::array<T, N>& a, const std::array<T, N>& b)
{
    static_assert(std::is_integral<T>::value && std::is_signed<T>::value,
                  "LocalInteraction cell indices must be signed integers");
    std::array<T, N> result;
    for (std::size_t d = 0; d != N; ++d)
    {
        if ((b[d] > 0 && a[d] < std::numeric_limits<T>::min() + b[d]) ||
            (b[d] < 0 && a[d] > std::numeric_limits<T>::max() + b[d]))
            throw std::invalid_argument("LocalInteraction: cell subtraction overflows");
        result[d] = a[d] - b[d];
    }
    return result;
}

template<typename TA>
void record_width(std::map<TA, std::size_t>& widths, const TA& atom,
                  std::size_t width, const char* name)
{
    const auto inserted = widths.insert(std::make_pair(atom, width));
    if (!inserted.second && inserted.first->second != width)
        throw std::invalid_argument(std::string("LocalInteraction: inconsistent ") + name);
}
} // namespace detail

/*! Rectangular bilinear CVC. The returned shape is {nl,nl,nr,nr}. */
template<typename Tleft, typename Tmetric, typename Tright>
Tensor<typename std::common_type<Tleft, Tmetric, Tright>::type>
contract_pair_vertices(const Tensor<Tleft>& left, const Tensor<Tmetric>& metric,
                       const Tensor<Tright>& right,
                       std::size_t n_left, std::size_t n_right)
{
    typedef typename std::common_type<Tleft, Tmetric, Tright>::type T;
    const std::size_t nl = detail::pair_count(n_left);
    const std::size_t nr = detail::pair_count(n_right);
    detail::product(nl, nr);
    detail::validate(left, 2, "left pair vertex");
    detail::validate(right, 2, "right pair vertex");
    detail::validate(metric, 2, "interaction metric");
    if (left.shape[0] != nl || right.shape[0] != nr ||
        left.shape[1] != metric.shape[0] || right.shape[1] != metric.shape[1])
        throw std::invalid_argument("LocalInteraction: incompatible CVC dimensions");
    const Tensor<T> l = detail::promote<T>(left);
    const Tensor<T> r = detail::promote<T>(right);
    const Tensor<T> v = detail::promote<T>(metric);
    const Tensor<T> lv = Tensor_Multiply::x0y1_x0a_ay1(l, v);
    const Tensor<T> result = Tensor_Multiply::x0y0_x0a_y0a(lv, r).reshape(
        {n_left, n_left, n_right, n_right});
    detail::validate(result, 4, "contracted local interaction");
    return result;
}

/*! One Fourier-q contribution in the ordered-density convention:
 * I(ab,cd;q) = sum conj(Q_left(ba,mu;q)) V(mu,nu;q) Q_right(cd,nu;q).
 * With real auxiliary functions Q_ab(-q)=conj(Q_ba(q)); this is the same
 * bilinear integral as above. No q weight or q sum is applied here.
 */
template<typename Tleft, typename Tmetric, typename Tright>
Tensor<typename std::common_type<Tleft, Tmetric, Tright>::type>
contract_ordered_pair_vertices(const Tensor<Tleft>& left,
                              const Tensor<Tmetric>& metric,
                              const Tensor<Tright>& right,
                              std::size_t n_left, std::size_t n_right)
{
    detail::validate(left, 2, "left pair vertex");
    if (left.shape[0] != detail::pair_count(n_left))
        throw std::invalid_argument("LocalInteraction: incompatible left pair count");
    Tensor<Tleft> reversed(left.shape);
    for (std::size_t a = 0; a != n_left; ++a)
        for (std::size_t b = 0; b != n_left; ++b)
            for (std::size_t mu = 0; mu != left.shape[1]; ++mu)
                reversed(a * n_left + b, mu) =
                    detail::conjugate(left(b * n_left + a, mu));
    return contract_pair_vertices(reversed, metric, right, n_left, n_right);
}

template<typename T>
struct DirectExchange
{
    Tensor<T> direct;   //!< U(a,b) = I(a,a,b,b).
    Tensor<T> exchange; //!< J(a,b) = I(a,b,b,a), without a Fock minus sign.
};

/*! Preserve the tensor's scalar type: no real-part projection or averaging. */
template<typename T>
DirectExchange<T> extract_direct_exchange(const Tensor<T>& integrals)
{
    detail::validate(integrals, 4, "local interaction tensor");
    const std::size_t n = integrals.shape[0];
    for (std::size_t d = 1; d != 4; ++d)
        if (integrals.shape[d] != n)
            throw std::invalid_argument("LocalInteraction: U/J requires one square orbital space");
    DirectExchange<T> result;
    result.direct = Tensor<T>({n, n});
    result.exchange = Tensor<T>({n, n});
    for (std::size_t a = 0; a != n; ++a)
        for (std::size_t b = 0; b != n; ++b)
        {
            result.direct(a, b) = integrals(a, a, b, b);
            result.exchange(a, b) = integrals(a, b, b, a);
        }
    return result;
}

/*! Build physical-cell pair vertices from raw FIRST-END LRI coefficients.
 *
 * Cs[I][{J,R}](mu,i,j) stores the I-auxiliary end only, not a complete fitted
 * AO product. orbitals[I][A](i,a) is the coefficient of the localized orbital
 * a on spatial AO (I,i) in the physical cell A. The output is
 *
 * Q[I][A](ab,mu) = sum_{J,R,i,j} Cs[I][{J,R}](mu,i,j)
 *   [conj(t[I,A](i,a))*t[J,A+R](j,b)
 *    +conj(t[J,A+R](j,a))*t[I,A](i,b)].
 *
 * Both LRI endpoints/AO orderings are explicit; they are not replaced by 2C.
 * Missing orbital cells or Cs blocks mean zero. Cells are physical signed
 * integer translations: there is no modulo wrapping, image replication or
 * implicit Fourier transform. The host supplies the complete desired orbital
 * support and coefficient blocks; no KS/DMFT projector is chosen here.
 */
template<typename TA, typename Tcell, std::size_t Ndim,
         typename Tcoefficient, typename Torbital>
std::map<TA, std::map<std::array<Tcell, Ndim>,
                     Tensor<typename std::common_type<Tcoefficient, Torbital>::type>>>
build_lri_pair_vertices(
    const std::map<TA, std::map<std::pair<TA, std::array<Tcell, Ndim>>,
                               Tensor<Tcoefficient>>>& coefficients,
    const std::map<TA, std::map<std::array<Tcell, Ndim>, Tensor<Torbital>>>& orbitals,
    std::size_t n_local)
{
    typedef typename std::common_type<Tcoefficient, Torbital>::type T;
    typedef std::array<Tcell, Ndim> Cell;
    std::map<TA, std::map<Cell, Tensor<T>>> result;
    const std::size_t pairs = detail::pair_count(n_local);
    std::map<TA, std::size_t> ao_widths, aux_widths;
    std::size_t orbital_blocks = 0, coefficient_blocks = 0;
    if (coefficients.empty() || orbitals.empty())
        throw std::invalid_argument("LocalInteraction: empty LRI or orbital input");
    for (const auto& atom : orbitals)
        for (const auto& cell : atom.second)
        {
            ++orbital_blocks;
            detail::validate(cell.second, 2, "local orbital coefficients");
            if (cell.second.shape[1] != n_local)
                throw std::invalid_argument("LocalInteraction: inconsistent local orbital count");
            detail::record_width(ao_widths, atom.first, cell.second.shape[0], "AO width");
        }
    for (const auto& atom : coefficients)
        for (const auto& block : atom.second)
        {
            ++coefficient_blocks;
            detail::validate(block.second, 3, "first-end LRI coefficients");
            detail::record_width(aux_widths, atom.first, block.second.shape[0], "auxiliary width");
            detail::record_width(ao_widths, atom.first, block.second.shape[1], "AO width");
            detail::record_width(ao_widths, block.first.first, block.second.shape[2], "AO width");
        }
    if (!orbital_blocks || !coefficient_blocks)
        throw std::invalid_argument("LocalInteraction: empty LRI or orbital block set");

    for (const auto& atom : coefficients)
    {
        const auto oi = orbitals.find(atom.first);
        if (oi == orbitals.end()) continue;
        for (const auto& block : atom.second)
        {
            const auto oj = orbitals.find(block.first.first);
            if (oj == orbitals.end()) continue;
            const Tensor<Tcoefficient>& c = block.second;
            const std::size_t ao_pairs = detail::product(c.shape[1], c.shape[2]);
            if (ao_pairs > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("LocalInteraction: AO-pair block exceeds BLAS integer range");
            detail::product(pairs, ao_pairs);
            const Tensor<T> c_flat = detail::promote<T>(c).reshape({c.shape[0], ao_pairs});
            for (const auto& cell_i : oi->second)
            {
                const auto cell_j = oj->second.find(detail::add(cell_i.first, block.first.second));
                if (cell_j == oj->second.end()) continue;
                const Tensor<Torbital>& ti = cell_i.second;
                const Tensor<Torbital>& tj = cell_j->second;
                Tensor<T>& q = result[atom.first][cell_i.first];
                if (q.empty()) q = Tensor<T>({pairs, c.shape[0]});
                // Form local pair weights once, then contract all auxiliary
                // indices through the existing GEMM helper. No AO ERI is built.
                Tensor<T> pair_weights({pairs, ao_pairs});
                for (std::size_t i = 0; i != c.shape[1]; ++i)
                    for (std::size_t j = 0; j != c.shape[2]; ++j)
                        for (std::size_t a = 0; a != n_local; ++a)
                            for (std::size_t b = 0; b != n_local; ++b)
                            {
                                pair_weights(a * n_local + b, i * c.shape[2] + j) =
                                    detail::conjugate(static_cast<T>(ti(i, a))) * static_cast<T>(tj(j, b)) +
                                    detail::conjugate(static_cast<T>(tj(j, a))) * static_cast<T>(ti(i, b));
                            }
                q += Tensor_Multiply::x0y0_x0a_y0a(pair_weights, c_flat);
            }
        }
    }
    for (const auto& atom : result)
        for (const auto& cell : atom.second)
            detail::validate(cell.second, 2, "built local pair vertex");
    return result;
}

/*! Explicit-cell local integral; missing directed metric blocks mean zero.
 * metric[I][{J,L}] = V_{I0,JL}; hence between cells A,B use L=B-A.
 * The caller must supply BOTH directed blocks when both are nonzero. No
 * Hermitian completion, BvK folding, Gamma sum or image sum is inferred.
 * Maps must be locally complete; MPI distribution/reduction remains with the
 * host, so this pure kernel can also be used without MPI initialization.
 */
template<typename TA, typename Tcell, std::size_t Ndim,
         typename Tleft, typename Tmetric, typename Tright>
Tensor<typename std::common_type<Tleft, Tmetric, Tright>::type>
contract_real_space_pair_vertices(
    const std::map<TA, std::map<std::array<Tcell, Ndim>, Tensor<Tleft>>>& left,
    const std::map<TA, std::map<std::pair<TA, std::array<Tcell, Ndim>>, Tensor<Tmetric>>>& metric,
    const std::map<TA, std::map<std::array<Tcell, Ndim>, Tensor<Tright>>>& right,
    std::size_t n_left, std::size_t n_right)
{
    typedef typename std::common_type<Tleft, Tmetric, Tright>::type T;
    const std::size_t nl = detail::pair_count(n_left);
    const std::size_t nr = detail::pair_count(n_right);
    detail::product(nl, nr);
    for (const auto& atom : left)
        for (const auto& cell : atom.second)
        {
            detail::validate(cell.second, 2, "left real-space vertex");
            if (cell.second.shape[0] != nl)
                throw std::invalid_argument("LocalInteraction: invalid left pair count");
        }
    for (const auto& atom : right)
        for (const auto& cell : atom.second)
        {
            detail::validate(cell.second, 2, "right real-space vertex");
            if (cell.second.shape[0] != nr)
                throw std::invalid_argument("LocalInteraction: invalid right pair count");
        }
    for (const auto& atom : metric)
        for (const auto& block : atom.second)
            detail::validate(block.second, 2, "real-space interaction metric");
    Tensor<T> result({n_left, n_left, n_right, n_right});
    for (const auto& atom_i : left)
    {
        const auto vi = metric.find(atom_i.first);
        if (vi == metric.end()) continue;
        for (const auto& atom_j : right)
            for (const auto& cell_i : atom_i.second)
                for (const auto& cell_j : atom_j.second)
                {
                    const auto block = vi->second.find(std::make_pair(
                        atom_j.first, detail::subtract(cell_j.first, cell_i.first)));
                    if (block == vi->second.end()) continue;
                    result += contract_pair_vertices(cell_i.second, block->second,
                                                     cell_j.second, n_left, n_right);
                }
    }
    detail::validate(result, 4, "real-space local interaction");
    return result;
}

template<typename TA, typename Tcell, std::size_t Ndim,
         typename Tvertex, typename Tmetric>
Tensor<typename std::common_type<Tvertex, Tmetric>::type>
contract_real_space_pair_vertices(
    const std::map<TA, std::map<std::array<Tcell, Ndim>, Tensor<Tvertex>>>& vertices,
    const std::map<TA, std::map<std::pair<TA, std::array<Tcell, Ndim>>, Tensor<Tmetric>>>& metric,
    std::size_t n_local)
{
    return contract_real_space_pair_vertices(vertices, metric, vertices, n_local, n_local);
}

} // namespace Local_Interaction
} // namespace RI
