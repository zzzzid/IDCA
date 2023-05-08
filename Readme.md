# Compiler Auto-tuning through Multiple Phase Learning

In this repository, we provide our code and the data.


## Dataset

The folder `cBench` contains all the programs.


## Algorithm

The folder `algorithm` contains all the code we used.

 `boca.py` is the BOCA algorithm.

 `CompTuner.py` is the CompTuner algorithm.

 `rio.py` is the Random Iteration Optimization algorithm.

 `OpenTuner.py` is the OpenTuner algorithm.

 `IDCA.py` is our proposed algorithm

We add a README file in the `algorithm` folder to help you understand and run the programs.

## Result

The folder `results` contains all the results of (speedup and time consumption for five techniques).

We add a README file in the `result` folder to help you understand the source data and result data for our expriments.

## Run

- In order to tune `gcc`'s optimization for program `benchmarks/cbench/automotive_bitcount`, execute the following command:

```
python3 runCompTuner.py --bin-path (your gcc location) --driver (your gcc driver) --linker (your gcc linker) --src-dir 'cbench/automotive_bitcount' --execute-params 20
```

## Note

Different versions of compilers use different compilation commands, so please pay attention to modify the relevant statements.