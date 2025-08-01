FROM ann-benchmarks

# RUN apt update && apt install -y cmake ninja-build
# 
# RUN git clone https://github.com/Visual-Computing/DynamicExplorationGraph.git; cd DynamicExplorationGraph; git checkout 5ff0a1b89395578a1a658a652e5d5914b45fb520
# 
# RUN python -m pip install pybind11 numpy setuptools>=77.0 build
# RUN cd DynamicExplorationGraph/python; python setup.py copy_build_files
# RUN cd DynamicExplorationGraph/python; python -m pip install .

RUN python -m pip install deglib==0.1.4

RUN python -c 'import deglib'

