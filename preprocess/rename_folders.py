import os
import argparse

def removeString(str_in, str_remove):
    """
    Removes a specified substring from an input string and converts the result to lowercase.

    Parameters:
    ----------
    str_in : str
        Input string.
    str_remove : str
        Substring to be removed from the input string.

    Returns:
    -------
    str
        Resulting string with the specified substring removed.
    """
    for char in str_remove:
        str_in = str_in.replace(char, "")
    str_in = str_in.lower()
    return str_in

def splitString(str_in, key, verbose=False):
    """
    Splits an input string based on a specified key and returns the last part.

    Parameters:
    ----------
    str_in : str
        Input string to be split.
    key : str
        Key used to split the string.
    verbose : bool, optional
        Enables/disables debug output (default is False).

    Returns:
    -------
    str
        The last part of the string after the split operation.
    """
    str_out = str_in.split(key)[-1].lower()
    if verbose:
        print("Split text:", str_out)
    return str_out

def run(path_data):
    """
    Renames subdirectories within a given root directory by removing specific substrings from their names.

    Parameters:
    ----------
    path_data : str
        Path to the root directory containing the subdirectories to be renamed.

    Notes:
    -----
    This function iterates through all subdirectories in the specified path, splits their names based on
    a defined key, and renames them to the result of the split if it differs from the original name.
    """
    for dirpath, subdirs, _ in os.walk(path_data):
        new_subdirs = [splitString(subdir, '.') for subdir in subdirs]
        for old_name, new_name in zip(subdirs, new_subdirs):
            src = os.path.join(dirpath, old_name)
            dst = os.path.join(dirpath, new_name)
            if src != dst:  # Avoid renaming if names are identical
                os.rename(src, dst)
                print(f"Renamed '{src}' to '{dst}'")

if __name__ == "__main__":
    # Argument parser configuration
    parser = argparse.ArgumentParser(description="Rename subdirectories in a specified path.")
    parser.add_argument(
        "--path_data", 
        type=str, 
        default="BD/Cropped Pollen Grains",  # Setting the default path
        help="Path to the root directory. Default is 'BD/Cropped Pollen Grains'."
    )

    # Parse arguments and execute the renaming function
    args = parser.parse_args()
    run(args.path_data)
