3ee83ca05b3e0f4fb738ba2880e479013531915cbbca29a67ce9f4d504a98d70  pnextract.exe
e24bdadf7cab857372d260c7d5e014bae02525c8f6d6309fb2d9990b16554eb2  pnflow.exe
7f445d74b756c9106a60d750acd1a2a87031fa2b6392bfbedabfa2a68e0727ed  voxelImageProcess.exe


To validate above .exe files in Windows PowerShell, run:

    certutil -hashfile   c:path/to/file.exe   SHA256

and compare the output with the above SHA256 checksums.

Or in Linux run from the bin/ directory:

    sha256sum -c checksums.sha256.md

