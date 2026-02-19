#!/usr/bin/env python3
"""
Export Poetry dependencies from pyproject.toml to requirements.txt files.
This version properly handles core dependencies and avoids duplication.
"""

import toml
from pathlib import Path
from typing import Union


def load_pyproject():
    """Load and parse pyproject.toml"""
    with open("pyproject.toml", "r") as f:
        return toml.load(f)


def format_dependency(name: str, spec: Union[str, dict]) -> str:
    """Format a dependency for requirements.txt"""
    if isinstance(spec, str):
        if spec == "*":
            return name
        elif spec.startswith("^"):
            # Poetry caret to pip format: ^5.2.0 means >=5.2.0,<6.0.0
            version = spec[1:]
            return f"{name}>={version}"
        elif spec[0] in "<>!=~":
            # Already a PEP 440 specifier (e.g. >=1.2.3)
            return f"{name}{spec}"
        else:
            return f"{name}=={spec}"
    elif isinstance(spec, dict):
        if "version" in spec:
            version = spec["version"]
            if version == "*":
                return name
            elif version.startswith("^"):
                return f"{name}>={version[1:]}"
            else:
                return f"{name}=={version}"
        elif "extras" in spec:
            extras = (
                ",".join(spec["extras"])
                if isinstance(spec["extras"], list)
                else spec["extras"]
            )
            version = spec.get("version", "")
            if version and version != "*":
                if version.startswith("^"):
                    return f"{name}[{extras}]>={version[1:]}"
                return f"{name}[{extras}]=={version}"
            return f"{name}[{extras}]"
    return name


def get_all_groups(pyproject: dict) -> dict:
    """Get all dependency groups including core"""
    groups = {}

    if "tool" in pyproject and "poetry" in pyproject["tool"]:
        poetry_config = pyproject["tool"]["poetry"]

        # Get core dependencies (for common.txt)
        if "dependencies" in pyproject["project"]:
            core_deps = {}
            for dep in pyproject["project"]["dependencies"]:
                # Skip python version specification
                dep_name, dep_spec = dep.split(" ", 1)
                if dep_name.lower() != "python":
                    core_deps[dep_name] = dep_spec
            groups["core"] = core_deps

        # Get group dependencies
        if "group" in poetry_config:
            for group_name, group_config in poetry_config["group"].items():
                if "dependencies" in group_config:
                    groups[group_name] = group_config["dependencies"]

    return groups


def main():
    """Main export function"""

    # Load pyproject.toml
    try:
        pyproject = load_pyproject()
    except FileNotFoundError:
        print("❌ Error: pyproject.toml not found in current directory")
        print("   Please run this script from your project root directory")
        return 1
    except toml.TomlDecodeError as e:
        print(f"❌ Error parsing pyproject.toml: {e}")
        return 1

    groups = get_all_groups(pyproject)

    # Create output directories
    req_dir = Path("face_detection_and_extraction/requirements")
    req_dir.mkdir(parents=True, exist_ok=True)

    similar_dir = Path("similar_face_filtering")
    similar_dir.mkdir(exist_ok=True)

    similar_test_dir = similar_dir / "tests"
    similar_test_dir.mkdir(exist_ok=True)

    face_test_dir = Path("face_detection_and_extraction/tests")
    face_test_dir.mkdir(parents=True, exist_ok=True)

    # Track core dependencies to avoid duplication
    core_deps = groups.get("core", {})
    core_deps_formatted = set()

    # 1. Export common.txt (core dependencies)
    print("\n📦 Exporting core dependencies...")
    with open(req_dir / "common.txt", "w") as f:
        for dep_name, dep_spec in sorted(core_deps.items()):
            formatted = format_dependency(dep_name, dep_spec)
            core_deps_formatted.add(
                formatted.split("==")[0].split(">=")[0]
            )  # Get base package name
            f.write(f"{dep_name} {dep_spec}\n")
    print(f"✓ common.txt ({len(core_deps)} dependencies)")

    # 2. Export dev.txt
    with open(req_dir / "dev.txt", "w") as f:
        for dep_name, dep_spec in sorted(groups.get("dev", {}).items()):
            f.write(f"{format_dependency(dep_name, dep_spec)}\n")
    print("✓ dev.txt")

    # 3. Export backend requirements
    backends = {
        "blazeface.txt": {
            "groups": ["pytorch", "onnx", "blazeface"],
            "include_common": True,
        },
        "mobile_facenet.txt": {
            "groups": ["pytorch", "onnx", "mobile-facenet"],
            "include_common": True,
        },
        "mtcnn.txt": {"groups": ["tensorflow", "mtcnn"], "include_common": True},
        "opencv.txt": {
            "groups": [],  # Only needs common deps
            "include_common": True,
        },
        "openvino.txt": {"groups": ["openvino"], "include_common": True},
        "yolov5-face.txt": {
            "groups": ["pytorch", "onnx", "yolov5-face"],
            "include_common": True,
        },
    }

    print("\n📦 Exporting backend requirements...")
    for filename, config in backends.items():
        deps = {}

        # Collect dependencies from specified groups
        for group_name in config.get("groups", []):
            if group_name in groups:
                deps.update(groups[group_name])

        # Write the file
        with open(req_dir / filename, "w") as f:
            # Add reference to common.txt if needed
            if config.get("include_common", False):
                f.write("-r common.txt\n")

            # Write dependencies, excluding those in common.txt if including common
            written_count = 0
            for dep_name, dep_spec in sorted(deps.items()):
                formatted = format_dependency(dep_name, dep_spec)
                base_name = formatted.split("==")[0].split(">=")[0].split("[")[0]

                # Skip if already in common.txt and we're including common
                if config.get("include_common") and base_name in core_deps_formatted:
                    continue

                f.write(f"{formatted}\n")
                written_count += 1

        print(f"✓ {filename} ({written_count} unique dependencies)")

    # 4. Export similar_face_filtering/requirements.txt
    print("\n📦 Exporting similar face filtering requirements...")
    with open(similar_dir / "requirements.txt", "w") as f:
        similar_deps = set()

        # Add tensorflow from tensorflow group
        for dep_name, dep_spec in groups.get("tensorflow", {}).items():
            similar_deps.add(format_dependency(dep_name, dep_spec))

        # Add dependencies from similar-face group
        for dep_name, dep_spec in groups.get("similar-face", {}).items():
            similar_deps.add(format_dependency(dep_name, dep_spec))

        # Add numpy (already in core but needed explicitly here)
        similar_deps.add("numpy")

        for dep in sorted(similar_deps):
            f.write(f"{dep}\n")
    print("✓ similar_face_filtering/requirements.txt")

    # 5. Export test requirements for both test directories
    # similar_face_filtering/tests/requirements-dev.txt (full dev deps)
    with open(face_test_dir / "requirements-test.txt", "w") as f:
        for dep_name, dep_spec in sorted(groups.get("test", {}).items()):
            f.write(f"{format_dependency(dep_name, dep_spec)}\n")
    # similar_face_filtering/tests/requirements-dev.txt (full dev deps)
    with open(similar_test_dir / "requirements-test.txt", "w") as f:
        for dep_name, dep_spec in sorted(groups.get("test", {}).items()):
            f.write(f"{format_dependency(dep_name, dep_spec)}\n")
    print("✓ similar_face_filtering/tests/requirements-dev.txt")

    # 6. Create main requirements.txt
    print("\n📦 Creating main requirements.txt...")
    with open("requirements.txt", "w") as f:
        f.write("-r face_detection_and_extraction/requirements/common.txt\n")
        f.write("-r face_detection_and_extraction/requirements/blazeface.txt\n")
        f.write("-r face_detection_and_extraction/requirements/mobile_facenet.txt\n")
        f.write("-r face_detection_and_extraction/requirements/mtcnn.txt\n")
        f.write("-r face_detection_and_extraction/requirements/opencv.txt\n")
        f.write("-r face_detection_and_extraction/requirements/openvino.txt\n")
        f.write("-r face_detection_and_extraction/requirements/yolov5-face.txt\n")
        f.write("-r face_detection_and_extraction/requirements/dev.txt\n")
        f.write("-r similar_face_filtering/requirements.txt\n")
        f.write(
            "# -r face_detection_and_extraction/requirements/trt_server.txt # Can cause error with nvidia-pyindex\n"
        )
    print("✓ requirements.txt")

    print("\n✅ All requirements files exported successfully!")
    print("\n📝 Usage examples:")
    print("   pip install -r requirements.txt  # Install everything")
    print(
        "   pip install -r face_detection_and_extraction/requirements/blazeface.txt  # Just blazeface"
    )
    print(
        "   pip install -r face_detection_and_extraction/requirements/common.txt  # Just core deps"
    )
    print("\n🔄 To keep in sync:")
    print("   Run this script after updating pyproject.toml")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
