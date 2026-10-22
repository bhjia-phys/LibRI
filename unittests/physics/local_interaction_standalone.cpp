#include "RI/physics/LocalInteraction.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace LI = RI::Local_Interaction;
typedef std::complex<double> Complex;
typedef std::array<int, 3> Cell;
typedef std::pair<int, Cell> AtomCell;
template<typename T>
using LriMap = std::map<int, std::map<AtomCell, RI::Tensor<T>>>;
template<typename T>
using CellMap = std::map<int, std::map<Cell, RI::Tensor<T>>>;

namespace
{
const Cell zero = {{0, 0, 0}};

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void close(Complex observed, Complex expected, const char* message)
{
    if (std::abs(observed - expected) > 2.0e-11 * (1.0 + std::abs(expected)))
    {
        std::cerr << message << ": observed " << observed << ", expected " << expected << '\n';
        throw std::runtime_error(message);
    }
}

template<typename Function>
void rejects(Function function, const char* message)
{
    bool rejected = false;
    try { function(); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, message);
}

// Compare the production factorization against a small explicitly formed AO
// four-index tensor and a separate four-leg complex orbital transformation.
void explicit_ao_control()
{
    const int nao = 3, nlocal = 2, naux = 3;
    const int nw[2] = {2, 1}, na[2] = {2, 1}, offset[2] = {0, 2};
    LriMap<double> cs, metric;
    CellMap<Complex> orbitals;
    RI::Tensor<Complex> t({nao, nlocal});
    const double pi = std::acos(-1.0);
    for (int i = 0; i != nao; ++i)
        for (int a = 0; a != nlocal; ++a)
            t(i, a) = std::polar(1.0 / std::sqrt(3.0),
                                 2.0 * pi * i * a / 3.0 + 0.19 * i * i);
    for (int a = 0; a != nlocal; ++a)
        for (int b = 0; b != nlocal; ++b)
        {
            Complex overlap;
            for (int i = 0; i != nao; ++i) overlap += std::conj(t(i, a)) * t(i, b);
            close(overlap, double(a == b), "test orbitals are not orthonormal");
        }
    for (int atom = 0; atom != 2; ++atom)
    {
        orbitals[atom][zero] = RI::Tensor<Complex>({std::size_t(nw[atom]), nlocal});
        for (int i = 0; i != nw[atom]; ++i)
            for (int a = 0; a != nlocal; ++a)
                orbitals[atom][zero](i, a) = t(offset[atom] + i, a);
        for (int other = 0; other != 2; ++other)
        {
            RI::Tensor<double>& c = cs[atom][AtomCell(other, zero)];
            c = RI::Tensor<double>({std::size_t(na[atom]), std::size_t(nw[atom]),
                                    std::size_t(nw[other])});
            for (int mu = 0; mu != na[atom]; ++mu)
                for (int i = 0; i != nw[atom]; ++i)
                    for (int j = 0; j != nw[other]; ++j)
                        c(mu, i, j) = 0.04 * (1 + 2 * atom + 3 * other + mu + i) - 0.013 * j;
        }
    }
    RI::Tensor<double> v({naux, naux});
    for (int p = 0; p != naux; ++p)
        for (int q = 0; q != naux; ++q)
        {
            v(p, q) = double(p == q);
            for (int k = 0; k != naux; ++k)
                v(p, q) += (0.2 + 0.07 * p - 0.03 * k) * (0.2 + 0.07 * q - 0.03 * k);
        }
    for (int i = 0; i != 2; ++i)
        for (int j = 0; j != 2; ++j)
        {
            RI::Tensor<double>& block = metric[i][AtomCell(j, zero)];
            block = RI::Tensor<double>({std::size_t(na[i]), std::size_t(na[j])});
            for (int p = 0; p != na[i]; ++p)
                for (int q = 0; q != na[j]; ++q)
                    block(p, q) = v(offset[i] + p, offset[j] + q);
        }

    // Complete AO product coefficients, including both atomic endpoints.
    RI::Tensor<double> product({naux, nao, nao});
    for (int atom = 0; atom != 2; ++atom)
        for (int other = 0; other != 2; ++other)
            for (int mu = 0; mu != na[atom]; ++mu)
                for (int i = 0; i != nw[atom]; ++i)
                    for (int j = 0; j != nw[other]; ++j)
                    {
                        const double c = cs.at(atom).at(AtomCell(other, zero))(mu, i, j);
                        product(offset[atom] + mu, offset[atom] + i, offset[other] + j) += c;
                        product(offset[atom] + mu, offset[other] + j, offset[atom] + i) += c;
                    }
    RI::Tensor<double> ao_eri({nao, nao, nao, nao});
    for (int i = 0; i != nao; ++i)
        for (int j = 0; j != nao; ++j)
            for (int k = 0; k != nao; ++k)
                for (int l = 0; l != nao; ++l)
                    for (int p = 0; p != naux; ++p)
                        for (int q = 0; q != naux; ++q)
                            ao_eri(i, j, k, l) += product(p, i, j) * v(p, q) * product(q, k, l);
    const auto vertices = LI::build_lri_pair_vertices(cs, orbitals, nlocal);
    const auto integrals = LI::contract_real_space_pair_vertices(vertices, metric, nlocal);
    const auto uj = LI::extract_direct_exchange(integrals);
    for (int a = 0; a != nlocal; ++a)
        for (int b = 0; b != nlocal; ++b)
            for (int c = 0; c != nlocal; ++c)
                for (int d = 0; d != nlocal; ++d)
                {
                    Complex expected;
                    for (int i = 0; i != nao; ++i)
                        for (int j = 0; j != nao; ++j)
                            for (int k = 0; k != nao; ++k)
                                for (int l = 0; l != nao; ++l)
                                    expected += std::conj(t(i, a)) * t(j, b) *
                                                std::conj(t(k, c)) * t(l, d) * ao_eri(i, j, k, l);
                    close(integrals(a, b, c, d), expected, "explicit AO ERI transform");
                    close(integrals(a, b, c, d), integrals(c, d, a, b), "pair symmetry");
                    close(integrals(a, b, c, d), std::conj(integrals(b, a, d, c)), "conjugation symmetry");
                }
    for (int a = 0; a != nlocal; ++a)
        for (int b = 0; b != nlocal; ++b)
        {
            close(uj.direct(a, b), integrals(a, a, b, b), "direct extraction");
            close(uj.exchange(a, b), integrals(a, b, b, a), "exchange extraction");
            require(std::abs(uj.exchange(a, b).imag()) < 1.e-12 &&
                        uj.exchange(a, b).real() >= -1.e-12, "exchange must be nonnegative");
            close(uj.exchange(a, a), uj.direct(a, a), "J_aa equals U_aa");
        }
    std::cout << "PASS explicit AO ERI / complex orthonormal orbitals / unequal atoms\n";
}

void two_cell_control()
{
    const Cell one = {{1, 0, 0}}, minus_one = {{-1, 0, 0}};
    LriMap<double> cs, v;
    CellMap<Complex> orbitals;
    cs[0][AtomCell(0, zero)] = RI::Tensor<double>({1, 1, 1});
    cs[0][AtomCell(0, zero)](0, 0, 0) = 0.5;
    const double theta = 0.43, phase = 0.61;
    orbitals[0][zero] = RI::Tensor<Complex>({1, 2});
    orbitals[0][one] = RI::Tensor<Complex>({1, 2});
    orbitals[0][zero](0, 0) = std::cos(theta);
    orbitals[0][zero](0, 1) = std::polar(std::sin(theta), phase);
    orbitals[0][one](0, 0) = -std::polar(std::sin(theta), -phase);
    orbitals[0][one](0, 1) = std::cos(theta);
    for (const Cell& cell : {zero, one, minus_one})
    {
        v[0][AtomCell(0, cell)] = RI::Tensor<double>({1, 1});
        v[0][AtomCell(0, cell)](0, 0) = cell == zero ? 2.0 : 0.5;
    }
    const auto vertices = LI::build_lri_pair_vertices(cs, orbitals, 2);
    const auto actual = LI::contract_real_space_pair_vertices(vertices, v, 2);
    RI::Tensor<Complex> expected({2, 2, 2, 2});
    const Cell cells[2] = {zero, one};
    for (int a = 0; a != 2; ++a)
        for (int b = 0; b != 2; ++b)
            for (int c = 0; c != 2; ++c)
                for (int d = 0; d != 2; ++d)
                    for (int i = 0; i != 2; ++i)
                        for (int j = 0; j != 2; ++j)
                        {
                            const auto& ti = orbitals.at(0).at(cells[i]);
                            const auto& tj = orbitals.at(0).at(cells[j]);
                            expected(a, b, c, d) += std::conj(ti(0, a)) * ti(0, b) *
                                (i == j ? 2.0 : 0.5) * std::conj(tj(0, c)) * tj(0, d);
                        }
    RI::Tensor<Complex> q0({4, 1}), qpi({4, 1});
    for (int pair = 0; pair != 4; ++pair)
    {
        q0(pair, 0) = vertices.at(0).at(zero)(pair, 0) + vertices.at(0).at(one)(pair, 0);
        qpi(pair, 0) = vertices.at(0).at(zero)(pair, 0) - vertices.at(0).at(one)(pair, 0);
    }
    RI::Tensor<double> v0({1, 1}), vpi({1, 1});
    v0(0, 0) = 2.5;
    vpi(0, 0) = 1.5;
    const auto full_q = Complex(0.5) * (LI::contract_ordered_pair_vertices(q0, v0, q0, 2, 2) +
                                LI::contract_ordered_pair_vertices(qpi, vpi, qpi, 2, 2));
    double wrong_gram_error = 0.0;
    const auto wrong_gram = Complex(0.5) * (LI::contract_pair_vertices(q0.conjugate(), v0, q0, 2, 2) +
                                   LI::contract_pair_vertices(qpi.conjugate(), vpi, qpi, 2, 2));
    for (std::size_t i = 0; i != expected.data->size(); ++i)
    {
        close(actual.ptr()[i], expected.ptr()[i], "two-cell direct quadrature");
        close(full_q.ptr()[i], expected.ptr()[i], "two-cell full-q / real-space agreement");
        wrong_gram_error = std::max(wrong_gram_error, std::abs(wrong_gram.ptr()[i] - expected.ptr()[i]));
    }
    require(wrong_gram_error > 0.1, "complex control must distinguish wrong Hermitian Gram");

    // A single-cell local density has I=2, whereas V(Gamma)=2.5 sums images.
    CellMap<double> localized;
    localized[0][zero] = RI::Tensor<double>({1, 1});
    localized[0][zero](0, 0) = 1.0;
    close(LI::contract_real_space_pair_vertices(localized, v, 1)(0, 0, 0, 0), 2.0,
          "onsite integral must not use Gamma image sum");
    close(0.5 * (v0(0, 0) + vpi(0, 0)), 2.0, "full-q normalization");

    // Cross-cell first-end coefficients must couple t(A) with t(A+R).
    LriMap<double> cross_cs;
    cross_cs[0][AtomCell(0, one)] = RI::Tensor<double>({1, 1, 1});
    cross_cs[0][AtomCell(0, one)](0, 0, 0) = 0.3;
    const auto cross = LI::build_lri_pair_vertices(cross_cs, orbitals, 2);
    require(cross.at(0).size() == 1, "finite support must not wrap or replicate cells");
    for (int a = 0; a != 2; ++a)
        for (int b = 0; b != 2; ++b)
            close(cross.at(0).at(zero)(2 * a + b, 0),
                  0.3 * (std::conj(orbitals.at(0).at(zero)(0, a)) * orbitals.at(0).at(one)(0, b) +
                         std::conj(orbitals.at(0).at(one)(0, a)) * orbitals.at(0).at(zero)(0, b)),
                  "cross-cell LRI endpoints");
    std::cout << "PASS two-cell full-q / direct cells / complex pair order / Gamma distinction\n";
}

void rectangular_and_validation_control()
{
    RI::Tensor<double> left({4, 2});
    RI::Tensor<Complex> right({9, 3}), metric({2, 3});
    for (std::size_t i = 0; i != left.data->size(); ++i) left.ptr()[i] = 0.1 * (i + 1);
    for (std::size_t i = 0; i != right.data->size(); ++i) right.ptr()[i] = Complex(0.05 * i, -0.03 * (i + 1));
    for (std::size_t i = 0; i != metric.data->size(); ++i) metric.ptr()[i] = Complex(0.04 * (i + 1), 0.02 * i);
    const auto output = LI::contract_pair_vertices(left, metric, right, 2, 3);
    static_assert(std::is_same<decltype(output), const RI::Tensor<Complex>>::value,
                  "mixed real/complex contraction must stay complex");
    for (int a = 0; a != 2; ++a)
        for (int b = 0; b != 2; ++b)
            for (int c = 0; c != 3; ++c)
                for (int d = 0; d != 3; ++d)
                {
                    Complex expected;
                    for (int p = 0; p != 2; ++p)
                        for (int q = 0; q != 3; ++q)
                            expected += left(2 * a + b, p) * metric(p, q) * right(3 * c + d, q);
                    close(output(a, b, c, d), expected, "rectangular mixed-type contraction");
                }
    rejects([&]() { LI::contract_pair_vertices(left, metric, right, 0, 3); }, "zero orbitals");
    rejects([&]() { LI::contract_pair_vertices(left, metric, right, 3, 3); }, "wrong pair count");
    rejects([&]() { LI::extract_direct_exchange(output); }, "rectangular U/J must be rejected");
    auto bad = metric.copy();
    bad(0, 0) = Complex(std::numeric_limits<double>::quiet_NaN(), 0.0);
    rejects([&]() { LI::contract_pair_vertices(left, bad, right, 2, 3); }, "nonfinite metric");
    RI::Tensor<double> missing({4, 2});
    missing.data.reset();
    rejects([&]() { LI::contract_pair_vertices(missing, metric, right, 2, 3); }, "missing storage");
    RI::Tensor<double> scalar({1, 1});
    scalar(0, 0) = 2.0;
    close(LI::contract_pair_vertices(scalar, scalar, scalar, 1, 1)(0, 0, 0, 0), 8.0,
          "real scalar contraction");
    scalar(0, 0) = 1.e200;
    rejects([&]() { LI::contract_pair_vertices(scalar, scalar, scalar, 1, 1); },
            "overflowed contraction must be rejected");
    LriMap<double> cs;
    CellMap<Complex> orbitals;
    cs[0][AtomCell(0, zero)] = RI::Tensor<double>({1, 2, 2});
    orbitals[0][zero] = RI::Tensor<Complex>({1, 1});
    rejects([&]() { LI::build_lri_pair_vertices(cs, orbitals, 1); }, "inconsistent AO widths");
    cs.clear();
    cs[0];
    rejects([&]() { LI::build_lri_pair_vertices(cs, orbitals, 1); }, "empty inner block map");
    std::cout << "PASS rectangular/mixed types and invalid-input rejection\n";
}
} // namespace

int main()
{
    try
    {
        explicit_ao_control();
        two_cell_control();
        rectangular_and_validation_control();
        std::cout << "LocalInteraction standalone controls passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
