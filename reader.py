
#!/usr/bin/env python3

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ==========================================
# RESOURCE LIMITS
# ==========================================

CPU_PERCENT = 95
RAM_LIMIT = "12G"


def enforce_resource_limits():

    if "--_limited" in sys.argv:
        return

    systemd_run = shutil.which("systemd-run")

    if not systemd_run:
        raise RuntimeError("systemd-run unavailable.")

    command = [
        systemd_run,
        "--user",
        "--scope",
        "--quiet",
        "--",
        sys.executable,
        str(Path(__file__).resolve()),
        "--_limited",
        *sys.argv[1:]
    ]

    os.execvp(systemd_run, command)


# ==========================================
# ARGUMENT PARSING
# ==========================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Convert SPSS SAV files to XLSX, CSV and ODS."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input SPSS .sav file"
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory (default: current directory)"
    )

    parser.add_argument(
        "--_limited",
        action="store_true",
        help=argparse.SUPPRESS
    )

    return parser.parse_args()


# ==========================================
# CONVERSION
# ==========================================

def convert_sav(input_file, output_dir):

    import pandas as pd
    import pyreadstat

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}"
        )

    if input_file.suffix.lower() != ".sav":
        raise ValueError("Input must be an SPSS .sav file.")

    output_dir.mkdir(parents=True, exist_ok=True)

    filename = input_file.stem

    print(f"Reading: {input_file}")

    # Preserve original numeric values rather than
    # replacing them with SPSS value labels.
    df, metadata = pyreadstat.read_sav(
        str(input_file),
        apply_value_formats=False
    )

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    # --------------------------------------
    # CSV
    # --------------------------------------

    csv_path = output_dir / f"{filename}.csv"

    print(f"Writing CSV: {csv_path}")

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------
    # XLSX
    # --------------------------------------

    # xlsx_path = output_dir / f"{filename}.xlsx"

    # # Excel worksheet maximum: 1,048,576 rows,
    # # including the header.
    # if len(df) > 1_048_575:
    #     print(
    #         "WARNING: Dataset exceeds Excel's "
    #         "maximum worksheet row count. Skipping XLSX."
    #     )
    # else:
    #     print(f"Writing XLSX: {xlsx_path}")

    #     df.to_excel(
    #         xlsx_path,
    #         index=False,
    #         engine="openpyxl",
    #         sheet_name="Data"
    #     )

    # --------------------------------------
    # ODS
    # --------------------------------------

    # ods_path = output_dir / f"{filename}.ods"

    # print(f"Writing ODS: {ods_path}")

    # df.to_excel(
    #     ods_path,
    #     index=False,
    #     engine="odf",
    #     sheet_name="Data"
    # )

    print("\nConversion completed.")

    print(f"CSV:  {csv_path}")
    # print(f"XLSX: {xlsx_path}")
    # print(f"ODS:  {ods_path}")


# ==========================================
# MAIN
# ==========================================

def main():

    args = parse_arguments()

    # Apply limits before importing heavy libraries
    # or loading the dataset.
    if not args._limited:
        enforce_resource_limits()

    try:
        convert_sav(
            args.input.resolve(),
            args.output_dir.resolve()
        )

    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


# to run, do python3 reader.py datasets/<file> -o ./exports