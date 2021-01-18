import fnmatch
import os
import shutil
import pkg_resources

import yaml


def get_profile_path():
    """Get the default location for site profiles.

    """
    return os.path.join(os.getenv("HOME"), ".config", "gstlal")


def installed_profiles():
    return [f[:-4] for f in os.listdir(get_profile_path()) if fnmatch.fnmatch(f, '*.yml')]


def current_profile():
    current_profile = os.path.join(get_profile_path(), "current.txt")
    if not os.path.exists(current_profile):
        raise ValueError("no current profile selected")
    with open(current_profile, "r") as f:
        return f.read()


def load_profile(profile=None):
    if not profile:
        profile = current_profile()
    profile_loc = os.path.join(get_profile_path(), f'{profile}.yml')
    with open(profile_loc, "r") as f:
        return yaml.safe_load(f)


def install_profiles(args):
    """Install site profiles in a central config location.

    """
    os.makedirs(get_profile_path(), exist_ok=True)
    if args.profile:
        profiles = [f for f in args.profile if fnmatch.fnmatch(f, '*.yml')]
    else:
        files = pkg_resources.resource_listdir('gstlal.dags', 'profiles')
        profiles = [f for f in files if fnmatch.fnmatch(f, '*.yml')]
        profiles = [pkg_resources.resource_filename('gstlal.dags', os.path.join('profiles', f)) for f in profiles]
    for profile in profiles:
        shutil.copy2(profile, get_profile_path())


def list_profiles(args):
    """List currently installed site profiles.

    """
    print("installed site profiles:")
    print("\t" + " ".join(installed_profiles()))


def get_profile(args):
    """Display currently selected site profile.

    """
    try:
        current = current_profile()
    except ValueError:
        print("no current profile selected")
    else:
        print(f"current profile: {current}")


def set_profile(args):
    if args.profile not in installed_profiles():
        print("invalid profile selection, run gstlal_grid_profile list to get a list of valid profiles")
    with open(os.path.join(get_profile_path(), "current.txt"), "w") as f:
        f.write(args.profile)
