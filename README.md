# OSCAR

Code and data for the numerical study in *OR for AI That Does OR: Routing LLMs up the Escalator inside the OSCAR Framework*.

OSCAR is **O**ptimization modeling by **S**imulator, **C**oder, **A**nd **R**eviewer. A Simulator, built from the problem description and certified against recorded example solutions, checks every solution a formulation produces; a Reviewer turns its verdict into an instruction for the next attempt; and the Escalator policy decides which model makes that attempt.

## Layout

```
oscar/              the framework
problems/           one directory per problem: data, oracle, certification evidence
settings.txt        what to run: the two models, the problem, the repetitions
start.py            a guided menu: prompts, defaults, and how long a run will take
run_experiment.py   the same runs as a single command, for scripting and batches
```

## Problems

| directory | class | formulation | objective |
|---|---|---|---|
| `problems/colombi2017` | Routing | vehicle routing and TSP | maximise |
| `problems/mehrotra1996` | Graph | graph optimization | minimise |
| `problems/letelier2022` | Packing | packing and cutting stock | minimise |
| `problems/elci2022` | Scheduling | scheduling | minimise |
| `problems/buchheim2018` | Quadratic | quadratic optimization | minimise |

Tasks come from the FrontierOR benchmark and keep its author-year names.

| file | what it is |
|---|---|
| `desc.txt` | the problem description. The only statement of the problem any model sees. |
| `instance/tiny_instance.json` | the instance that is solved. |
| `gurobi_solution/tiny_solution.json` | its reference solution and optimal objective value. |
| `feasibility_check.py` | the trusted checker, used only for scoring. No model sees it and nothing it returns re-enters the loop. |
| `simulator_certified.py` | the certified Simulator used for this problem. |
| `solution_schema.json` | the decision syntax a formulation must emit. |

## Certification evidence

The Simulator is written by a language model from the description, so it is not trusted until it has been checked. `certification_examples.json` lists solutions whose correct verdict is already known -- feasible references with their objective values, and solutions that should be rejected, with the reason. A candidate Simulator is certified only when it reproduces every one of them.

| file | what it is |
|---|---|
| `certification_examples.json` | the examples: an instance, a solution, the expected verdict, and where applicable the objective value and the reason for rejection. |
| `partial_examples.json` | fragments of solutions showing what one individual rule allows. Used only while building the Simulator; never sent during a run. |
| `cert_*.json` | solutions the examples refer to that are not benchmark references. |
| `instance/`, `gurobi_solution/` | all six instances each task ships, with their reference solutions. The examples are recorded against these. |
| `simulator_build.json`, `simulator_build_log.json` | the build record: which candidate was produced, which examples it failed, what it was told, and the round on which it certified. |
| `simulator_build_metrics.jsonl` | the model calls the build made, per role, with tokens and cost. |

Only `tiny_instance.json` is ever solved. The five larger instances are present because the examples are recorded against them; `generate_simulator.py` and `robustness.py` in `oscar/` are the code that used them.

## Running

Requires Python 3.10+, `openai`, and `coptpy` with a licence.

```bash
pip install openai coptpy
```

### Setting up a run

`settings.txt` is the one place to say what to run. Edit it in any text editor; both entry points read it. Your API key does **not** go in it.

| setting | what it does | default |
|---|---|---|
| `small_model` | the cheap Coder, the structure constructor, and syntax repair | `qwen3.5-flash` |
| `large_model` | the Reviewer, the expensive Coder, and Simulator construction | `qwen3.6-flash` |
| `api_base` | the OpenAI-compatible endpoint your key belongs to | unset |
| `problem` | which of the five to run | `mehrotra1996` |
| `oneshot_reps` | repetitions of the one-shot baseline | `5` |
| `oneshot_model` | which of the two models writes the formulation: `small` or `large` | `small` |
| `oscar_runs` | independent OSCAR runs -- raise this for a success rate rather than one outcome | `1` |
| `oscar_n_per_level` | Algorithm 1's n_j: attempts allowed at each menu configuration before escalating. `1`, or four comma-separated numbers in cost order such as `1,1,3,3` | `1` |

**Two models, and exactly two.** OSCAR is a cheap model that tries first and a stronger one it escalates to; naming the same model twice leaves the Escalator nothing to escalate to, and `start.py` says so rather than running it.

Precedence runs weakest to strongest: `settings.txt`, then an environment variable, then a command-line flag. So `OSCAR_SMALL_MODEL` in your shell overrides the file, and `--model` overrides both.

There are three ways in. The first two run the same experiments and differ only in whether you are asked questions; both read `settings.txt`, so set that up first. The third is for changing the method rather than running it.

| | what it is for |
|---|---|
| `python start.py` | a menu. Prompts, defaults taken from `settings.txt`, and an estimate of how long the run will take before it starts |
| `python run_experiment.py --method oscar` | the same run as one command, so it can go in a shell loop, a batch left overnight, or a cluster job |
| `oscar/oscar_escalator.py`, `oscar/build_simulator.py` | the framework itself: a different menu of Reviewer/Coder configurations, a Simulator of your own |

Nothing in the second row is harder than the first -- it is seven flags -- it is simply not interactive.

### Guided

```bash
python start.py
```

A menu: check your setup, enter your key, run the one-shot baseline, run OSCAR, see what you have so far. Every prompt offers what `settings.txt` says, so pressing Enter runs exactly that; option 2 reloads the file after you have edited it. It prints the command it is about to run, so it also serves as a way to learn the flags, and it says what the run is likely to cost you in time before it starts.

Every run there uses the certified Simulator from the paper, so the numbers you get are comparable with the published ones. Building a Simulator of your own is possible and the code and data for it are included, but it is deliberately not on that menu: a different oracle quietly makes the results incomparable. See [Building a Simulator](#building-a-simulator).

Your key is held in memory only: never written to disk, never printed, and never placed on a command line where a process list would show it. It reaches each experiment through the environment and is discarded when you quit. If you would rather not type it at all, set `OSCAR_API_KEY` in your shell beforehand and the tool will use it without asking.

### As one command

`run_experiment.py` runs exactly what the menu runs, without asking anything. Give it a method and it takes everything else from `settings.txt`; every flag below overrides one line of that file.

```bash

export OSCAR_API_KEY=sk-...                      # your key. Not in settings.txt
export COPT_LICENSE_DIR=/path/to/copt            # without a licence COPT solves a
                                                 # smaller problem silently and
                                                 # every result is wrong

# exactly what settings.txt says
python run_experiment.py --method oscar

# ...or override any of it
python run_experiment.py --problem colombi2017 --method oscar --runs 10
python run_experiment.py --problem colombi2017 --method oneshot --reps 10

# one row of the paper's table: the baseline and OSCAR on the same problem
for p in mehrotra1996 elci2022; do
  python run_experiment.py --problem $p --method oneshot --reps 10
  python run_experiment.py --problem $p --method oscar   --runs 10
done
```

Asking for ten runs twice gives twenty: both methods number their output directories in order and continue after whatever is already there, so a job you interrupt can be picked up by repeating the command.

### Getting a key

**You need one.** No credentials ship with this package and every experiment calls a model. Three things have to agree: an OpenAI-compatible endpoint, a key that works on it, and two model names spelled the way that endpoint spells them.

The paper used Qwen, served by Alibaba Cloud. To follow it as closely as possible: open an account on Alibaba Cloud Model Studio (Bailian), issue an API key there, and copy the OpenAI-compatible base URL from that console -- at the time of writing `https://dashscope.aliyuncs.com/compatible-mode/v1` for Beijing or `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` for Singapore. Take it from the console rather than from here; these move. Our own runs went through a private workspace deployment, so your console may spell the models a little differently from `qwen3.5-flash` and `qwen3.6-flash`; use the names it gives you.

Any other OpenAI-compatible provider works the same way -- see [Using your own models](#using-your-own-models).

The endpoint and the model names go in `settings.txt`. **The key does not**: give it to `start.py` at the prompt, where it is held in memory only, or set it in your shell, where this package picks it up without asking.

```bash
export OSCAR_API_KEY=sk-...          # macOS, Linux
setx OSCAR_API_KEY sk-...            # Windows, then open a new terminal
```

`start.py` option 2, then 3, says all of this at the terminal, and option 1 will make one very short model call to confirm the key works -- worth doing before committing an hour to a run.

What does **not** need a key: importing the package, loading a certified Simulator, and scoring a solution that is already recorded. Those touch no model, which is how the setup check tells a missing key apart from a broken install. They are not a way to run an experiment.

### Using your own models

The study used two models from one family, a cheaper and a stronger one, reached through an OpenAI-compatible endpoint. Name your own in `settings.txt`:

```
small_model = gpt-4o-mini
large_model = gpt-4o
api_base    = https://api.openai.com/v1
```

or, for one run only, in the environment:

```bash
export OSCAR_SMALL_MODEL=gpt-4o-mini
export OSCAR_LARGE_MODEL=gpt-4o
```

Left alone, the defaults reproduce the paper's configuration. The Escalator's menu is built from these two, so naming them is enough; no source change is needed to run on a different provider. `--model` on the one-shot baseline overrides the Coder for that run alone.

Results are written under `results/<problem>/<method>/`: the attempts, the Simulator's verdicts, the final solution, and token and cost accounting per role.

### How long a run takes

Longer than it looks. Almost none of the time is this code: a run waits for the endpoint to answer each model call, and then for COPT to solve each program the Coder writes. Both are slow, and neither is under our control.

Median wall-clock per repetition and per run, over the runs reported in the paper, on one desktop machine against a commercial endpoint. These distributions have long tails, so the 90th percentile is given beside the median; they are what to expect, not a bound.

| problem | class | one-shot repetition | OSCAR run at n=1 | OSCAR reached the optimum |
|---|---|---|---|---|
| `colombi2017` | Routing | 8 min (p90 33) | 41 min (p90 64) | 72% |
| `mehrotra1996` | Graph | 2 min (p90 7) | 9 min (p90 12) | 100% |
| `letelier2022` | Packing | 6 min (p90 22) | 30 min (p90 137) | 97% |
| `elci2022` | Scheduling | 6 min (p90 22) | 30 min (p90 57) | 95% |
| `buchheim2018` | Quadratic | 9 min (p90 38) | 95 min (p90 183) | 100% |

**`mehrotra1996` is the one to start with**, and it is what `settings.txt` points at: OSCAR reached the optimum on all 100 runs, the median run took nine minutes, and it uses about a quarter of the tokens of any other problem here. Ten runs of it is an hour and a half, which is enough to see the method work.

The quadratic problem is the slow one, by a wide margin: its formulation is a dense quadratic program, and the solve, not the model, is what takes the time. A single OSCAR run there is usually over an hour and the slowest tenth past three; `oscar_runs = 10` on it is a day of machine time.

For the widest gap between the baseline and OSCAR, use `elci2022`: one-shot reaches the optimum on 4% of repetitions with the cheap model and 35% with the strong one, against 95% for OSCAR.

At `n_per_level = 1,1,3,3` a run may make eight attempts in a window instead of four, so budget about twice the figures above.

Nothing is lost if you stop a job: each repetition and each run is written out as it completes, `start.py` option 5 reports what is on disk, and starting again adds to it.

## Building a Simulator

Everything needed to build one from scratch is here, and Section 2 of the paper is what it implements. This is the part to reach for if you want to check the construction itself rather than the results that follow from it.

```bash
cd oscar
export OSCAR_PROBLEM=../problems/colombi2017

python build_simulator.py --check                       # re-certify the shipped one
python build_simulator.py --force --out ../mine.py      # build a new one
```

A problem directory must carry everything the build reads. `build_simulator.py` fails at the first missing file, naming it; these are the ones it looks for.

| file | required | what it is |
|---|---|---|
| `desc.txt` | yes | the problem statement, the only description the model sees |
| `certification_examples.json` | yes | the labelled examples the candidate must reproduce |
| `feasibility_check.py` | yes | the trusted checker, used by the robustness battery |
| `instance/` | yes | the instances the examples refer to |
| the solutions the examples name | yes | each example points at a solution file, by path |
| `problem_config.json` | when the solution schema is not the default | names the fields and objective sense |
| `solution_schema.json` | recommended | the declared solution fields, used to build malformed cases |
| `partial_examples.json` | optional | per-rule fragments, used only during the build |

The endpoint comes from `settings.txt`, the same file the rest of the packagereads, so there is nothing to configure twice. The API key does not live in
that file. If `OSCAR_API_KEY` is not already in the environment,`build_simulator.py` asks for it at the terminal and holds it in memory only,
as `start.py` does. Set it in your shell beforehand and it is picked up without asking.

The build writes `simulator_certified.py` beside those files, or wherever
`--out` points. Nothing else in the directory is read or written.

`--force` asks the strong model to write a Simulator from the description, then certifies it against that problem's recorded examples, retrying with the failures fed back until it reproduces every verdict or gives up. `generate_simulator.py` holds the build prompts and the certification loop; `robustness.py` is the generated-input battery a candidate has to survive.

A run uses the Simulator at `problems/<problem>/simulator_certified.py`, or whatever `OSCAR_SIMULATOR_PATH` points at. **Results from a Simulator you built are not comparable with the paper's**, since the oracle deciding what counts as an improvement is no longer the same one.

## Configuration

Two layers, and it is worth keeping them apart. `settings.txt` is what you choose. The table below is what is **held fixed**: the same five values for every problem, set in `UNIFORM` at the top of `run_experiment.py`, so that a number from one problem can be compared with a number from another. Changing them is a change to the experiment, not to how you run it.

| setting | value | meaning |
|---|---|---|
| `OSCAR_SOLVER_TL` | `0` | the Coder is not told to set a solver time limit |
| `OSCAR_SOLVER_TIME_CAP` | `0` | and none is imposed afterwards |
| `OSCAR_SOLVER_MEMLIMIT_MB` | `3072` | COPT memory limit, the same for every problem |
| `OSCAR_PROCESS_MEMLIMIT_MB` | `3456` | a guard on the whole process, for a program that exhausts memory before the solver starts |
| `--proc-tl` | `1800` | wall-clock bound on one generated program, in seconds |

The runs reported in the paper were not uniform in these two respects: two problems ran with a 240-second instructed solver limit and the rest with none, and one problem was given twice the solver memory of the others because its formulation needs about 2.2 GB. The single configuration here is the more permissive of each, so results from different problems are comparable.

## Not included

The scripts that construct the certification example sets: the examples themselves are here, which is what is needed to repeat the certification.

Four further problems with certified Simulators that the paper does not report, and the intermediate Simulator variants from each build history.
