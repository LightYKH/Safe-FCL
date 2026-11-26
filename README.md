# Safe-FCL: Continual Risk Awareness and Fine-Grained Safety Aggregation for Federated Continual Learning of LLMs

Safe-FCL is a safety framework for federated continual learning of large
language models.

- **Continual Risk Awareness (CRA)** learns risk patterns from local LoRA
  update trajectories. During risk detector adaptation, it projects parameter
  updates using protection bases derived from previous tasks.
- **Fine-Grained Safety Aggregation (FGSA)** combines client-level safety
  scores with module-level deviations from historical statistics to regulate
  the contribution of each client's LoRA modules to the global adapter.

![Overview of Safe-FCL, including Continual Risk Awareness and Fine-Grained Safety Aggregation](assets/safe-fcl-overview.png)

## Main Results

Global-model safety results under Protocol 1 for the primary task sequence
(Table 2 in the paper). Each stage average covers the tasks learned so far.
Bold and underlined averages mark the best and second-best scores at each stage.

<table>
  <thead>
    <tr>
      <th rowspan="2" scope="col" align="left">Method</th>
      <th rowspan="2" scope="col">Task</th>
      <th colspan="5" scope="colgroup">Rule (%)</th>
      <th colspan="5" scope="colgroup">MD-Judge (%)</th>
    </tr>
    <tr>
      <th scope="col">Task 1</th><th scope="col">Task 2</th><th scope="col">Task 3</th><th scope="col">Task 4</th><th scope="col">Avg.</th>
      <th scope="col">Task 1</th><th scope="col">Task 2</th><th scope="col">Task 3</th><th scope="col">Task 4</th><th scope="col">Avg.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" scope="rowgroup" align="left">FedAvg</th>
      <td>T1</td>
      <td align="right">46.62</td><td align="right">–</td><td align="right">–</td><td align="right">–</td><td align="right">46.62</td>
      <td align="right">41.26</td><td align="right">–</td><td align="right">–</td><td align="right">–</td><td align="right">41.26</td>
    </tr>
    <tr>
      <td>T2</td>
      <td align="right">46.85</td><td align="right">72.49</td><td align="right">–</td><td align="right">–</td><td align="right">59.67</td>
      <td align="right">41.72</td><td align="right">77.16</td><td align="right">–</td><td align="right">–</td><td align="right">59.44</td>
    </tr>
    <tr>
      <td>T3</td>
      <td align="right">36.60</td><td align="right">66.67</td><td align="right">33.33</td><td align="right">–</td><td align="right">45.53</td>
      <td align="right">33.57</td><td align="right">65.73</td><td align="right">29.37</td><td align="right">–</td><td align="right">42.89</td>
    </tr>
    <tr>
      <td>T4</td>
      <td align="right">43.12</td><td align="right">72.03</td><td align="right">42.19</td><td align="right">57.81</td><td align="right">53.79</td>
      <td align="right">40.33</td><td align="right">70.16</td><td align="right">38.93</td><td align="right">61.07</td><td align="right">52.62</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <th rowspan="4" scope="rowgroup" align="left">FLAME</th>
      <td>T1</td>
      <td align="right">64.34</td><td align="right">–</td><td align="right">–</td><td align="right">–</td><td align="right">64.34</td>
      <td align="right">59.21</td><td align="right">–</td><td align="right">–</td><td align="right">–</td><td align="right">59.21</td>
    </tr>
    <tr>
      <td>T2</td>
      <td align="right">60.37</td><td align="right">79.25</td><td align="right">–</td><td align="right">–</td><td align="right">69.81</td>
      <td align="right">56.88</td><td align="right">81.12</td><td align="right">–</td><td align="right">–</td><td align="right">69.00</td>
    </tr>
    <tr>
      <td>T3</td>
      <td align="right">46.15</td><td align="right">71.33</td><td align="right">36.13</td><td align="right">–</td><td align="right">51.20</td>
      <td align="right">41.49</td><td align="right">72.73</td><td align="right">32.40</td><td align="right">–</td><td align="right">48.87</td>
    </tr>
    <tr>
      <td>T4</td>
      <td align="right">46.62</td><td align="right">72.96</td><td align="right">44.99</td><td align="right">72.73</td><td align="right">59.33</td>
      <td align="right">42.19</td><td align="right">73.89</td><td align="right">40.09</td><td align="right">76.92</td><td align="right">58.27</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <th rowspan="4" scope="rowgroup" align="left">LASA</th>
      <td>T1</td>
      <td align="right">55.94</td><td align="right">–</td><td align="right">–</td><td align="right">–</td><td align="right">55.94</td>
      <td align="right">48.95</td><td align="right">–</td><td align="right">–</td><td align="right">–</td><td align="right">48.95</td>
    </tr>
    <tr>
      <td>T2</td>
      <td align="right">55.71</td><td align="right">78.32</td><td align="right">–</td><td align="right">–</td><td align="right">67.02</td>
      <td align="right">53.85</td><td align="right">82.98</td><td align="right">–</td><td align="right">–</td><td align="right">68.42</td>
    </tr>
    <tr>
      <td>T3</td>
      <td align="right">39.63</td><td align="right">74.36</td><td align="right">34.03</td><td align="right">–</td><td align="right">49.34</td>
      <td align="right">35.20</td><td align="right">74.36</td><td align="right">29.60</td><td align="right">–</td><td align="right">46.39</td>
    </tr>
    <tr>
      <td>T4</td>
      <td align="right">47.09</td><td align="right">77.16</td><td align="right">44.29</td><td align="right">61.77</td><td align="right">57.58</td>
      <td align="right">42.19</td><td align="right">77.86</td><td align="right">36.36</td><td align="right">66.43</td><td align="right">55.71</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <th rowspan="4" scope="rowgroup" align="left">Safe-FedLLM-Step</th>
      <td>T1</td>
      <td align="right">84.38</td><td align="right">–</td><td align="right">–</td><td align="right">–</td><td align="right"><ins>84.38</ins></td>
      <td align="right">85.31</td><td align="right">–</td><td align="right">–</td><td align="right">–</td><td align="right"><ins>85.31</ins></td>
    </tr>
    <tr>
      <td>T2</td>
      <td align="right">92.07</td><td align="right">87.88</td><td align="right">–</td><td align="right">–</td><td align="right">89.98</td>
      <td align="right">96.04</td><td align="right">90.91</td><td align="right">–</td><td align="right">–</td><td align="right">93.48</td>
    </tr>
    <tr>
      <td>T3</td>
      <td align="right">47.55</td><td align="right">76.92</td><td align="right">44.06</td><td align="right">–</td><td align="right">56.18</td>
      <td align="right">43.12</td><td align="right">77.62</td><td align="right">38.69</td><td align="right">–</td><td align="right">53.14</td>
    </tr>
    <tr>
      <td>T4</td>
      <td align="right">92.54</td><td align="right">90.68</td><td align="right">88.81</td><td align="right">91.14</td><td align="right">90.79</td>
      <td align="right">94.17</td><td align="right">92.77</td><td align="right">89.04</td><td align="right">98.60</td><td align="right">93.65</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <th rowspan="4" scope="rowgroup" align="left">Safe-FedLLM-Client</th>
      <td>T1</td>
      <td align="right">62.47</td><td align="right">–</td><td align="right">–</td><td align="right">–</td><td align="right">62.47</td>
      <td align="right">57.58</td><td align="right">–</td><td align="right">–</td><td align="right">–</td><td align="right">57.58</td>
    </tr>
    <tr>
      <td>T2</td>
      <td align="right">61.54</td><td align="right">82.52</td><td align="right">–</td><td align="right">–</td><td align="right">72.03</td>
      <td align="right">57.58</td><td align="right">85.78</td><td align="right">–</td><td align="right">–</td><td align="right">71.68</td>
    </tr>
    <tr>
      <td>T3</td>
      <td align="right">41.96</td><td align="right">73.43</td><td align="right">35.43</td><td align="right">–</td><td align="right">50.27</td>
      <td align="right">37.53</td><td align="right">73.43</td><td align="right">31.47</td><td align="right">–</td><td align="right">47.48</td>
    </tr>
    <tr>
      <td>T4</td>
      <td align="right">68.76</td><td align="right">83.45</td><td align="right">65.50</td><td align="right">80.19</td><td align="right">74.48</td>
      <td align="right">65.03</td><td align="right">85.78</td><td align="right">63.17</td><td align="right">85.78</td><td align="right">74.94</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <th rowspan="4" scope="rowgroup" align="left">Safe-FedLLM-Shadow</th>
      <td>T1</td>
      <td align="right">53.38</td><td align="right">–</td><td align="right">–</td><td align="right">–</td><td align="right">53.38</td>
      <td align="right">48.25</td><td align="right">–</td><td align="right">–</td><td align="right">–</td><td align="right">48.25</td>
    </tr>
    <tr>
      <td>T2</td>
      <td align="right">89.98</td><td align="right">91.14</td><td align="right">–</td><td align="right">–</td><td align="right"><ins>90.56</ins></td>
      <td align="right">96.04</td><td align="right">94.17</td><td align="right">–</td><td align="right">–</td><td align="right"><ins>95.11</ins></td>
    </tr>
    <tr>
      <td>T3</td>
      <td align="right">79.72</td><td align="right">90.91</td><td align="right">70.86</td><td align="right">–</td><td align="right"><ins>80.50</ins></td>
      <td align="right">84.38</td><td align="right">90.44</td><td align="right">75.29</td><td align="right">–</td><td align="right"><ins>83.37</ins></td>
    </tr>
    <tr>
      <td>T4</td>
      <td align="right">92.77</td><td align="right">93.47</td><td align="right">90.68</td><td align="right">91.61</td><td align="right"><ins>92.13</ins></td>
      <td align="right">93.94</td><td align="right">96.04</td><td align="right">92.54</td><td align="right">98.83</td><td align="right"><ins>95.34</ins></td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <th rowspan="4" scope="rowgroup" align="left">Safe-FCL (Ours)</th>
      <td>T1</td>
      <td align="right">90.44</td><td align="right">–</td><td align="right">–</td><td align="right">–</td><td align="right"><strong>90.44</strong></td>
      <td align="right">92.31</td><td align="right">–</td><td align="right">–</td><td align="right">–</td><td align="right"><strong>92.31</strong></td>
    </tr>
    <tr>
      <td>T2</td>
      <td align="right">94.41</td><td align="right">92.07</td><td align="right">–</td><td align="right">–</td><td align="right"><strong>93.24</strong></td>
      <td align="right">97.67</td><td align="right">95.57</td><td align="right">–</td><td align="right">–</td><td align="right"><strong>96.62</strong></td>
    </tr>
    <tr>
      <td>T3</td>
      <td align="right">85.55</td><td align="right">91.61</td><td align="right">79.25</td><td align="right">–</td><td align="right"><strong>85.47</strong></td>
      <td align="right">88.81</td><td align="right">91.84</td><td align="right">82.05</td><td align="right">–</td><td align="right"><strong>87.57</strong></td>
    </tr>
    <tr>
      <td>T4</td>
      <td align="right">93.71</td><td align="right">92.31</td><td align="right">92.31</td><td align="right">91.84</td><td align="right"><strong>92.54</strong></td>
      <td align="right">95.10</td><td align="right">95.10</td><td align="right">94.64</td><td align="right">99.07</td><td align="right"><strong>95.98</strong></td>
    </tr>
  </tbody>
</table>

## Quick Start

Use an isolated Python 3.10 environment and run the following commands from
the repository root. The example generates LoRA update features for two
simulated clients and checks CRA and FGSA with synthetic inputs.

```bash
export PYTHONDONTWRITEBYTECODE=1
python -m pip install -r requirements.txt
python scripts/validate_release.py
python -m unittest discover -s tests -v
python scripts/synthetic_demo.py
python scripts/materialize_code.py --method safe_fcl \
  --destination ../safe-fcl-inspect/code --task 1 --order order2
```

## Experiment Configuration

The [Safe-FCL configuration](configs/methods/safe_fcl.json) specifies training
and risk detector settings. The [FGSA configuration](configs/gates/safe_fcl.json)
specifies aggregation parameters. The command-line method identifier is
`safe_fcl`.

Experiment inputs must be prepared locally. The evaluation script reports
Rule and MD-Judge scores for a selected model checkpoint and target task.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `configs/` | Task sequences and Safe-FCL settings |
| `src/core/` | Federated fine-tuning, client risk assessment, and FGSA |
| `src/collection/` | Collection of local LoRA update trajectories from a fixed task reference |
| `src/probe_models/`, `src/probe_training/` | Risk detector models, feature preparation, and CRA adaptation |
| `src/ood/` | Federated fine-tuning under the Task 5 OOD protocol |
| `scripts/` | Code assembly, feature collection, risk detector training, and federated training entry points |
| `evaluation/` | Response generation and scoring with Rule and MD-Judge |
| `tests/` | CPU tests for the implementation and artifact compatibility |
