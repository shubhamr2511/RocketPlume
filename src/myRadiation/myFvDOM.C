/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Version:  v2112
-------------------------------------------------------------------------------
Description
    Non-gray fvDOM implementation — see myFvDOM.H for details.

References:
    Raithby & Chui, J. Heat Transfer, 1990.
    Modest, "Radiative Heat Transfer", 3rd ed., Ch. 16.
    HITEMP: Rothman et al., JQSRT 2010.

\*---------------------------------------------------------------------------*/

#include "myFvDOM.H"
#include "fvmDiv.H"
#include "fvmSup.H"
#include "mathematicalConstants.H"
#include "addToRunTimeSelectionTable.H"

namespace Foam
{
namespace radiation
{

defineTypeNameAndDebug(myFvDOM, 0);
addToRunTimeSelectionTable(radiationModel, myFvDOM, T);

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * //

void myFvDOM::buildQuadrature()
{
    // Build Sn (level-symmetric) discrete ordinate quadrature
    // For nPhi=4, nTheta=4: 64 total directions covering the full sphere
    // Using equal-angle subdivision of phi and theta

    nRay_ = 4 * nPhi_ * nTheta_;
    rayDir_.resize(nRay_);
    omega_.resize(nRay_);

    label rayI = 0;
    const scalar dPhi   = constant::mathematical::twoPi / (4 * nPhi_);
    const scalar dTheta = constant::mathematical::pi / nTheta_;

    for (label iPhi = 0; iPhi < 4 * nPhi_; iPhi++)
    {
        const scalar phi = (iPhi + 0.5) * dPhi;

        for (label iTheta = 0; iTheta < nTheta_; iTheta++)
        {
            const scalar theta = (iTheta + 0.5) * dTheta;

            rayDir_[rayI] = vector
            (
                Foam::sin(theta) * Foam::cos(phi),
                Foam::sin(theta) * Foam::sin(phi),
                Foam::cos(theta)
            );

            // Solid angle: dOmega = sin(theta)*dTheta*dPhi
            omega_[rayI] = Foam::sin(theta) * dTheta * dPhi;

            rayI++;
        }
    }

    Info << "myFvDOM: built quadrature with " << nRay_
         << " ray directions (nPhi=" << nPhi_
         << ", nTheta=" << nTheta_ << ")" << endl;
}


void myFvDOM::initialiseFields()
{
    ILambda_.resize(nBands_);

    for (label b = 0; b < nBands_; b++)
    {
        ILambda_[b].resize(nRay_);

        for (label d = 0; d < nRay_; d++)
        {
            const word fieldName =
                "ILambda_band" + Foam::name(b) + "_ray" + Foam::name(d);

            ILambda_[b].set
            (
                d,
                new volScalarField
                (
                    IOobject
                    (
                        fieldName,
                        mesh_.time().timeName(),
                        mesh_,
                        IOobject::READ_IF_PRESENT,
                        IOobject::AUTO_WRITE
                    ),
                    mesh_,
                    dimensionedScalar(fieldName, dimMass/pow3(dimTime), 0)
                    // Units: W/(m^2·sr) = kg/s^3
                )
            );
        }
    }
}


scalar myFvDOM::fractionalBlackbody(const scalar lambdaT) const
{
    // Approximation of fractional blackbody function F(0→λT)
    // F = fraction of total blackbody power emitted below wavelength λ at temperature T
    // lambdaT in µm·K
    // Chang & Rhee (1984) series, accurate to ~0.1% for lambdaT in [200, 10^6] µm·K

    if (lambdaT <= 0) return 0.0;

    // Constants for series expansion (Planck integral)
    const scalar C2 = 14387.77;  // µm·K  (hc/k)
    const scalar x  = C2 / lambdaT;

    if (x > 100) return 0.0;    // Practically zero emission at very short wavelengths

    // Sum terms: F = (15/pi^4) * sum_{n=1}^{N} e^{-nx}/n * (x^3 + 3x^2/n + 6x/n^2 + 6/n^3)
    scalar F = 0;
    const scalar pi4_over15 = Foam::pow(constant::mathematical::pi, 4) / 15.0;

    for (label n = 1; n <= 20; n++)
    {
        const scalar sn = scalar(n);
        const scalar en = Foam::exp(-sn * x);
        if (en < 1e-15) break;
        F += en / sn * (((x*x*x) + 3.0*x*x/sn + 6.0*x/(sn*sn) + 6.0/(sn*sn*sn)));
    }

    F *= 15.0 / (constant::mathematical::pi * constant::mathematical::pi *
                 constant::mathematical::pi * constant::mathematical::pi);

    return Foam::min(Foam::max(F, 0.0), 1.0);
}


scalar myFvDOM::planckBandIntegral
(
    const scalar Tcell,
    const scalar nuLow,
    const scalar nuHigh
) const
{
    // Total blackbody emissive power: Eb = sigma * T^4
    const scalar sigma  = 5.67037e-8;   // Stefan-Boltzmann [W/m^2/K^4]
    const scalar Eb     = sigma * Foam::pow4(Tcell);

    // Convert wavenumber [cm^-1] to wavelength [µm]:  lambda = 10000 / nu
    const scalar lambdaLow  = 10000.0 / nuHigh;   // note: low nu → high lambda
    const scalar lambdaHigh = 10000.0 / nuLow;

    const scalar F_high = fractionalBlackbody(lambdaHigh * Tcell);
    const scalar F_low  = fractionalBlackbody(lambdaLow  * Tcell);

    // Band emissive power per steradian: Ib_band = (F_high - F_low)*Eb / pi
    const scalar Ib_band = (F_high - F_low) * Eb / constant::mathematical::pi;

    return Foam::max(Ib_band, 0.0);
}


scalar myFvDOM::solveRay
(
    label bandI,
    label rayI,
    const volScalarField& kappa,
    const volScalarField& Ib_band
)
{
    volScalarField& I = ILambda_[bandI][rayI];
    const vector& dir = rayDir_[rayI];

    // Build surface normal flux: J = direction vector as surfaceScalarField
    // This is the convective flux for the transport equation
    surfaceScalarField Ji
    (
        IOobject("Ji", mesh_.time().timeName(), mesh_),
        mesh_,
        dimensionedScalar("Ji", dimless, 0)
    );

    // Dot ray direction with face area normals (upwind flux)
    const surfaceVectorField& Sf = mesh_.Sf();
    forAll(Ji, faceI)
    {
        Ji[faceI] = dir & Sf[faceI];
    }
    forAll(Ji.boundaryField(), patchI)
    {
        forAll(Ji.boundaryField()[patchI], faceI)
        {
            Ji.boundaryFieldRef()[patchI][faceI] =
                dir & Sf.boundaryField()[patchI][faceI];
        }
    }

    // Assemble RTE finite-volume equation:
    //   div(J_d * I) + kappa * I = kappa * Ib_band
    fvScalarMatrix IEqn
    (
        fvm::div(Ji, I)
      + fvm::Sp(kappa, I)
     ==
        kappa * Ib_band
    );

    IEqn.relax();
    const scalar residual = IEqn.solve().initialResidual();

    return residual;
}


// * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

myFvDOM::myFvDOM(const volScalarField& T, const dictionary& dict)
:
    radiationModel(typeName, T),
    nBands_(coeffs_.lookupOrDefault<label>("nBands", 1)),
    nPhi_(coeffs_.subDict("fvDOMCoeffs").lookupOrDefault<label>("nPhi", 4)),
    nTheta_(coeffs_.subDict("fvDOMCoeffs").lookupOrDefault<label>("nTheta", 4)),
    maxIter_(coeffs_.subDict("fvDOMCoeffs").lookupOrDefault<label>("maxIter", 10)),
    tolerance_(coeffs_.subDict("fvDOMCoeffs").lookupOrDefault<scalar>("tolerance", 1e-4)),
    T_(T)
{
    // Read band limits
    if (nBands_ > 1)
    {
        bandLimits_ = coeffs_.lookup("bandLimits");
        if (bandLimits_.size() != nBands_)
        {
            FatalErrorInFunction
                << "nBands = " << nBands_
                << " but bandLimits has " << bandLimits_.size() << " entries"
                << abort(FatalError);
        }
    }
    else
    {
        // Gray: single band covering full spectrum
        bandLimits_.resize(1);
        bandLimits_[0] = Pair<scalar>(1.0, 1e6);
    }

    buildQuadrature();
    initialiseFields();

    Info << "myFvDOM: initialised with " << nBands_ << " bands, "
         << nRay_ << " rays" << nl
         << "    Band limits [cm^-1]:" << nl;
    forAll(bandLimits_, b)
    {
        Info << "    Band " << b << ": ["
             << bandLimits_[b].first() << ", "
             << bandLimits_[b].second() << "]" << endl;
    }
}


// * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

myFvDOM::~myFvDOM()
{}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void myFvDOM::calculate()
{
    // Outer iteration loop over all bands
    for (label b = 0; b < nBands_; b++)
    {
        const scalar nuLow  = bandLimits_[b].first();
        const scalar nuHigh = bandLimits_[b].second();

        // Load band-specific absorption coefficient from registry
        const word kappaName = "kappaBand" + Foam::name(b);
        const volScalarField& kappa =
            mesh_.lookupObject<volScalarField>(kappaName);

        // Compute band Planck emission at each cell
        volScalarField Ib_band
        (
            IOobject("Ib_band" + Foam::name(b), mesh_.time().timeName(), mesh_),
            mesh_,
            dimensionedScalar("Ib", kappa.dimensions()*dimLength, 0)
        );

        forAll(T_, cellI)
        {
            Ib_band[cellI] = planckBandIntegral(T_[cellI], nuLow, nuHigh);
        }

        // Iterative solve over ray directions
        scalar maxResid = GREAT;
        label iter = 0;

        while (maxResid > tolerance_ && iter < maxIter_)
        {
            maxResid = 0;
            for (label d = 0; d < nRay_; d++)
            {
                scalar resid = solveRay(b, d, kappa, Ib_band);
                maxResid = Foam::max(maxResid, resid);
            }
            iter++;
        }

        Info << "myFvDOM: band " << b
             << " converged in " << iter << " iterations"
             << " (residual = " << maxResid << ")" << endl;
    }
}


tmp<volScalarField> myFvDOM::Rp() const
{
    // Radiation source term (positive = net emission > absorption)
    // Qr = sum_b [ kappa_b * (4*pi*Ib_b - G_b) ]
    // where G_b = sum_d (I_{b,d} * omega_d)

    tmp<volScalarField> tRp
    (
        new volScalarField
        (
            IOobject("Rp", mesh_.time().timeName(), mesh_),
            mesh_,
            dimensionedScalar("Rp", dimMass/dimLength/pow3(dimTime), 0)
        )
    );
    volScalarField& Rp_ = tRp.ref();

    for (label b = 0; b < nBands_; b++)
    {
        const word kappaName = "kappaBand" + Foam::name(b);
        const volScalarField& kappa =
            mesh_.lookupObject<volScalarField>(kappaName);

        const scalar nuLow  = bandLimits_[b].first();
        const scalar nuHigh = bandLimits_[b].second();

        forAll(mesh_.cells(), cellI)
        {
            const scalar Ib = planckBandIntegral(T_[cellI], nuLow, nuHigh);

            scalar G_b = 0;
            for (label d = 0; d < nRay_; d++)
            {
                G_b += ILambda_[b][d][cellI] * omega_[d];
            }

            Rp_[cellI] += kappa[cellI] *
                (4.0 * constant::mathematical::pi * Ib - G_b);
        }
    }

    return tRp;
}


tmp<volScalarField::Internal> myFvDOM::Ru() const
{
    return tmp<volScalarField::Internal>
    (
        new volScalarField::Internal
        (
            IOobject("Ru", mesh_.time().timeName(), mesh_),
            mesh_,
            dimensionedScalar("Ru", dimMass/dimLength/pow3(dimTime), 0)
        )
    );
}


tmp<volScalarField> myFvDOM::G() const
{
    tmp<volScalarField> tG
    (
        new volScalarField
        (
            IOobject("G", mesh_.time().timeName(), mesh_),
            mesh_,
            dimensionedScalar("G", dimMass/pow3(dimTime), 0)
        )
    );
    volScalarField& G_ = tG.ref();

    for (label b = 0; b < nBands_; b++)
    {
        for (label d = 0; d < nRay_; d++)
        {
            G_ += ILambda_[b][d] * omega_[d];
        }
    }

    return tG;
}


const volScalarField& myFvDOM::ILambda(label bandI, label rayI) const
{
    return ILambda_[bandI][rayI];
}


bool myFvDOM::read()
{
    if (radiationModel::read())
    {
        coeffs_.readIfPresent("nBands", nBands_);
        coeffs_.readIfPresent("bandLimits", bandLimits_);
        coeffs_.subDict("fvDOMCoeffs").readIfPresent("maxIter", maxIter_);
        coeffs_.subDict("fvDOMCoeffs").readIfPresent("tolerance", tolerance_);
        return true;
    }
    return false;
}


} // End namespace radiation
} // End namespace Foam

// ************************************************************************* //
