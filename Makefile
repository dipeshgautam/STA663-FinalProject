Report.pdf: Report.tex profiler.txt latex_tables/Runtimes.tex latex_tables/inverseMethods.tex figures/Original.png figures/Detected.png figures/kDistribution.png figures/Trace.png latex_tables/featuresDetected.tex
	pdflatex Report
	pdflatex Report
	pdflatex Report

latex_tables/featuresDetected.tex figures/Original.png figures/Detected.png figures/kDistribution.png figures/Trace.png: python_scripts/createPlots.py Data/chainZ.npy Data/chainK.npy Data/chainSigmaX.npy Data/chainSigmaA.npy Data/chainAlpha.npy Data/SimulatedData.npy Data/ZOriginal.npy Data/AOriginal.npy
	python python_scripts/createPlots.py

latex_tables/inverseMethods.tex: python_scripts/compareInverse.py
	python python_scripts/compareInverse.py

latex_tables/Runtimes.tex: python_scripts/compareSampler.py Data/SimulatedData.npy python_scripts/Cython_functions.so
	python python_scripts/compareSampler.py
    
Data/chainZ.npy Data/chainK.npy Data/chainSigmaX.npy Data/chainSigmaA.npy Data/chainAlpha.npy: Data/SimulatedData.npy python_scripts/finalsimulation.py
	python python_scripts/finalsimulation.py

Cython_functions.so:
	cd python_scripts ; python Cython_setup.py build_ext --inplace; cd ..

Data/SimulatedData.npy Data/ZOriginal.npy Data/AOriginal.npy: python_scripts/GenerateData.py
	python python_scripts/GenerateData.py
    
    
all: Report.pdf
