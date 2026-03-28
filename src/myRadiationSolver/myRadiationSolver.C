/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Version:  v2112
-------------------------------------------------------------------------------
Application
    myRadiationSolver

Description
    Standalone radiation-only solver for IR signature estimation from an
    aircraft exhaust plume.

    The solver reads prescribed T, p, XH2O, XCO2, XCO, and kappaBand{b}
    fields (written by Python scripts from HITEMP data), then solves the
    non-gray Radiative Transfer Equation using the modified fvDOM model
    (myFvDOM) for each spectral band.

    No flow equations are solved — radiation is computed on a frozen
    (prescribed) flow field.

Usage
    myRadiationSolver [OPTIONS]

    Parallel:
        decomposePar
        mpirun -np 4 myRadiationSolver -parallel
        reconstructPar

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "myFvDOM.H"
#include "simpleControl.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Standalone non-gray RTE solver for aircraft plume IR signature.\n"
        "Reads prescribed T/p/species/kappa fields, solves multi-band fvDOM."
    );

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    simpleControl simple(mesh);

    // ---- Read prescribed flow fields ----

    Info << "Reading temperature field T\n" << endl;
    volScalarField T
    (
        IOobject("T", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE),
        mesh
    );

    Info << "Reading pressure field p\n" << endl;
    volScalarField p
    (
        IOobject("p", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE),
        mesh
    );

    Info << "Reading species mole fractions\n" << endl;
    volScalarField XH2O
    (
        IOobject("XH2O", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE),
        mesh
    );
    volScalarField XCO2
    (
        IOobject("XCO2", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE),
        mesh
    );
    volScalarField XCO
    (
        IOobject("XCO", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE),
        mesh
    );

    // Read radiation properties and determine nBands
    const dictionary& radDict = mesh.lookupObject<IOdictionary>("radiationProperties");
    const label nBands = radDict.lookupOrDefault<label>("nBands", 1);

    Info << "Reading " << nBands << " absorption coefficient band fields\n" << endl;

    // kappaBand fields must be present in 0/ (written by map_plume_data.py)
    PtrList<volScalarField> kappaBands(nBands);
    for (label b = 0; b < nBands; b++)
    {
        const word name = "kappaBand" + Foam::name(b);
        kappaBands.set
        (
            b,
            new volScalarField
            (
                IOobject(name, runTime.timeName(), mesh,
                         IOobject::MUST_READ, IOobject::AUTO_WRITE),
                mesh
            )
        );
    }

    // ---- Initialise radiation model ----
    Info << "Initialising myFvDOM radiation model\n" << endl;
    radiation::myFvDOM radiation(T, radDict);

    // ---- Time loop (single pseudo-step for steady RTE) ----
    while (simple.loop())
    {
        Info << "Time = " << runTime.timeName() << nl << endl;

        // Solve the multi-band RTE
        radiation.calculate();

        // Compute and report total incident radiation G
        volScalarField G = radiation.G();
        G.write();

        // Compute net radiation source term
        volScalarField Qr = radiation.Rp();
        Qr.write();

        Info << "Radiation solve complete." << nl
             << "    max(G) = " << max(G).value() << " W/m^2" << nl
             << "    max(|Qr|) = " << max(mag(Qr)).value() << " W/m^3" << endl;

        runTime.write();
    }

    Info << nl << "Solver finished. Post-process with:" << nl
         << "    paraFoam                        -- visualise fields" << nl
         << "    python scripts/extract_ir.py    -- sensor power per band" << nl
         << endl;

    return 0;
}

// ************************************************************************* //
