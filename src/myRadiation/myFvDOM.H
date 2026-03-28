/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Version:  v2112
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Description
    Non-gray finite-volume Discrete Ordinates Method (fvDOM) radiation solver
    extended to support multi-band spectral radiation using HITEMP-derived
    absorption coefficients.

    Key additions over the base fvDOM:
      - nBands_: number of spectral bands
      - bandLimits_: wavenumber range [cm^-1] for each band
      - Intensity fields indexed by [band][ray]: ILambda_band{b}_ray{d}
      - Band Planck function integration: planckBandFraction()
      - Reads kappaBand{b} volScalarField for each band

    RTE solved per band per ray:
      div(J_d, I_{b,d}) + kappa_b * I_{b,d} = kappa_b * Ib_band(T)

\*---------------------------------------------------------------------------*/

#ifndef myFvDOM_H
#define myFvDOM_H

#include "radiationModel.H"
#include "volFields.H"
#include "fvMatrices.H"
#include "Pair.H"

namespace Foam
{
namespace radiation
{

/*---------------------------------------------------------------------------*\
                         Class myFvDOM Declaration
\*---------------------------------------------------------------------------*/

class myFvDOM
:
    public radiationModel
{
    // Private data

        //- Number of spectral bands
        label nBands_;

        //- Band limits [cm^-1]: List of (nuLow, nuHigh) pairs
        List<Pair<scalar>> bandLimits_;

        //- Number of phi (azimuthal) angles
        label nPhi_;

        //- Number of theta (polar) angles
        label nTheta_;

        //- Total number of ray directions
        label nRay_;

        //- Maximum iterations per radiation solve
        label maxIter_;

        //- Convergence tolerance
        scalar tolerance_;

        //- Intensity fields: ILambda_band{b}_ray{d}
        //  Outer list: bands; inner list: ray directions
        List<PtrList<volScalarField>> ILambda_;

        //- Cached reference to temperature field
        const volScalarField& T_;

        //- Discrete ordinate direction vectors
        List<vector> rayDir_;

        //- Solid angle weights for each ray [sr]
        List<scalar> omega_;


    // Private member functions

        //- Build S4 discrete ordinate quadrature set
        void buildQuadrature();

        //- Initialise intensity fields for all bands and rays
        void initialiseFields();

        //- Planck function blackbody intensity at temperature T [W/(m^2 sr)]
        //  Integrated over a spectral band [nuLow, nuHigh] cm^-1
        //  Uses fractional blackbody function: Ib_band = f(T,nuHigh)*sigma*T^4/pi
        //  - f: fraction of total blackbody emission in [0, nu]
        scalar planckBandIntegral
        (
            const scalar T,
            const scalar nuLow,
            const scalar nuHigh
        ) const;

        //- Fractional blackbody function F(lambda*T)
        //  Fraction of total emission in [0, lambda] for given lambda*T product
        //  Uses Chang-Rhee series approximation (accurate to 0.1%)
        scalar fractionalBlackbody(const scalar lambdaT) const;

        //- Solve RTE for one (band, ray) pair
        //  Returns residual
        scalar solveRay
        (
            label bandI,
            label rayI,
            const volScalarField& kappa,
            const volScalarField& Ib_band
        );


public:

    //- Runtime type information
    TypeName("myFvDOM");


    // Constructors

        myFvDOM(const volScalarField& T, const dictionary& dict);

        //- Disallow default bitwise copy construct
        myFvDOM(const myFvDOM&) = delete;

        //- Disallow default bitwise assignment
        void operator=(const myFvDOM&) = delete;


    //- Destructor
    virtual ~myFvDOM();


    // Member Functions

        //- Main radiation solve: loops over all bands and rays
        virtual void calculate();

        //- Return radiative heat source term [W/m^3]
        //  Sum over all bands: Qr = sum_b [ 4*pi*kappa_b*Ib_b - kappa_b*G_b ]
        virtual tmp<volScalarField> Rp() const;

        virtual tmp<volScalarField::Internal> Ru() const;

        //- Return total incident radiation G [W/m^2]
        //  G = sum_b sum_d (I_{b,d} * omega_d)
        tmp<volScalarField> G() const;

        //- Return intensity field for given band and ray
        const volScalarField& ILambda(label bandI, label rayI) const;

        //- Number of bands
        label nBands() const { return nBands_; }

        //- Number of rays
        label nRay() const { return nRay_; }

        //- Ray direction vectors
        const List<vector>& rayDir() const { return rayDir_; }

        //- Solid angle weights
        const List<scalar>& omega() const { return omega_; }

        virtual bool read();
};


} // End namespace radiation
} // End namespace Foam

#endif

// ************************************************************************* //
