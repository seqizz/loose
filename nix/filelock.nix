{
  lib,
  buildPythonPackage,
  fetchPypi,
  hatch-vcs,
  hatchling,
}:
# nixpkgs ships 3.20.3, which is below the security-fixed floor declared in
# pyproject.toml. Vendored here until nixpkgs catches up; drop this file and
# use python.pkgs.filelock once it is >= 4.2.1.
buildPythonPackage rec {
  pname = "filelock";
  version = "4.2.4";
  pyproject = true;

  src = fetchPypi {
    inherit pname version;
    # Matches the sdist hash uv.lock resolved for this version
    hash = "sha256-xthmNTXhZQ7kMFB7T/use0ZN1kt2R+oBEqn0BkmdFj0=";
  };

  build-system = [
    hatch-vcs
    hatchling
  ];

  # Upstream test suite pulls pytest-asyncio/-timeout and a virtualenv path that
  # has to be excluded; the import check is enough for a lock library
  doCheck = false;
  pythonImportsCheck = [ "filelock" ];

  meta = {
    changelog = "https://github.com/tox-dev/py-filelock/releases/tag/${version}";
    description = "Platform independent file lock for Python";
    homepage = "https://github.com/tox-dev/py-filelock";
    license = lib.licenses.unlicense;
    maintainers = [ "seqizz" ];
    platforms = lib.platforms.linux;
  };
}
