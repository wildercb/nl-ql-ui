from setuptools import setup, find_packages

setup(
    name="mppw-mcp",
    version="0.1.0",
    description="MPP Watson MCP unified architecture",
    packages=find_packages(exclude=("venv*", "tests*", "docs*")),
    include_package_data=True,
    python_requires=">=3.9",
) 