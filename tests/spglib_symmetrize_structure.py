def symmetrize_structure(structure_path, symprec=1e-2):
    """
    Determine space group with spglib.
    Write dataset to file with _spglib suffix.
    Symmetrize to primitive cell compatible with QE pw.x ibrav=0.
    structure_path : str
        Path to the structure file.
    symmetrize : bool
        If True, symmetrize atomic positions and lattice vectors using spglib
        and overwrite the structure file.
    symprec : float
        Symmetry precision for spglib. Default is 1e-5.
    """
    atoms = read(structure_path)
    # Detect symmetry with spglib
    lattice = atoms.get_cell()
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()
    cell_tuple = (lattice, positions, numbers)
    dataset = spglib.get_symmetry_dataset(cell_tuple, symprec=symprec)
    spacegroup = dataset.international
    number = dataset.number
    print(f"Detected space group: {spacegroup} ({number})")
    # Write spglib dataset to file
    structure_stem = os.path.splitext(structure_path)[0]
    spglib_path = f"{structure_stem}_spglib"
    with open(spglib_path, 'w') as f:
        f.write(str(dataset))
    # Symmetrize structure
    symmetrized = spglib.standardize_cell(
        cell_tuple,
        to_primitive=True,
        no_idealize=False,
        symprec=symprec)
    std_lattice, std_positions, std_numbers = symmetrized
    # Convert to pymatgen structure
    symmetrized_pmg = Structure(
        lattice=std_lattice,
        species=std_numbers,
        coords=std_positions,
        coords_are_cartesian=False)
    # Write filename
    path = Path(structure_path)
    name = path.stem
    prefix = "sym-"
    if len(name) > 13 and name[12] == '-' and name[:12].isdigit():
        serial_tag = name[:13]
        rest = name[13:]
        new_name = f"{serial_tag}{prefix}{rest}"
    else:
        new_name = f"{prefix}{name}"
    symmetrized_path = Path.cwd() / f"{new_name}.cif"
    # Write with pymatgen
    writer = CifWriter(symmetrized_pmg, symprec=None, refine_struct=False)
    writer.write_file(str(symmetrized_path))
    print(f"Symmetrized structure saved to: {symmetrized_path}")

    return symmetrized_path
