from setuptools import setup, find_packages

setup(
    name="toolrank",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["httpx>=0.27.0"],
    description="ToolRank Tracker SDK — implicit tool call tracking for AI agents",
)
