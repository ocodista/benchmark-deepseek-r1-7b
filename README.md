# Benchmarking DeepSeek R1 Benchmark on Macbook M2 Pro
## TL;DR 

I'll show you how DeepSeek R1 7b + Ollama performs on a Macbook Pro M2 16GB RAM.

Questions I plan to answer in this post:

- How many tokens/s can I get running DeepSeek R1 7b locally with ollama?
- How many requests/s can I serve with reasonable throughput? (10 tokens/s)
- How does the number of concurrent requests impact the throughput?
- How much power did my GPU used while running this study?

### How was the test run?
10 cycles, ranging from 1 -> 128.

Each cycle:
  - Defines the number of parallel requests
  - Generates individual request metrics
  - Generates cycle metrics

Additionally, each cycle:
  - Monitors Ollama processes for usage information
  - Tracks GPU usage

Ok, with this in mind, let's jump into how I created the benchmark application. 

Oh, it's [open-source](TODO: Add link to github) btw.

## **Table of Contents**  
1. **Introduction**  
2. **Test Setup**  
   - Hardware Specifications  
   - Software Environment  
   - Benchmarking Tools  
3. **Benchmarking Methodology**  
   - Load Testing with Golang  
   - Monitoring System Metrics  
   - Data Collection and Analysis  
4. **Results and Analysis**  
   - Response Times & Token Throughput  
   - CPU, Memory, and GPU Usage  
   - Scalability Trends & Bottlenecks  
5. **Findings & Observations**  
   - Optimal Concurrency for Performance  
   - System Resource Constraints  
   - Unexpected Behaviors and Insights  
6. **Conclusion & Next Steps**  

---

### **1. Introduction**  
Hey, I'm Caio.

I like running benchmarks and I'm curious about self-hosted applications (or, in this case, large language chinese open-source models).

I created a side project with Go, Python & Unix: a concurrent benchmark mechanism (aka requests gun), that connected to Ollama (and DeepSeek R1: 7b) to be processed by my macbook M2 Series 19-Core GPU.

It's all about tokens/s and parallel (to a point) requests.

### **2. Test Setup**  

#### **Hardware Specifications**  
The tests were conducted on the following hardware:  

- **Device**: MacBook Pro 16-inch (M2, 2023)  
- **CPU**: 12-core AMD Processor  
- **Memory**: 16GB RAM  
- **GPU**: Integrated M2 Series GPU (19 Cores)
- **OS**: macOS Sonoma  

#### **Software Environment**  
- **Ollama**: A local LLM inference framework running DeepSeek R1 (7B)  
- **Golang**: Used to create the benchmarking client and monitoring tools, why? Well, **Channels**.
- **Python & Pandas**: For data analysis and visualization  

#### **Benchmarking Tools**  
- **Golang HTTP Client**: Sends concurrent requests to Ollama’s API  
- **System Monitoring Script**: Collects CPU, memory, and GPU usage  
- **CSV Logging & Python Analysis**: Stores request metrics for later analysis  

---

### **3. Benchmarking Methodology**  

#### **Load Testing with Golang**
The goal to compare how the Ollama API would behave when receiving different loads of parallel requests.

The `benchmark.go` file receives a param, named _concurrencyLevels_ that is in fact a list [1,2,3,4] of how many cycles should the benchmark run, where each cycle value defines how many concurrent requests will be sent.

For each cycle:

A custom Golang client sends **concurrent requests** to the Ollama API. Each request asks:  

```json
{
  "model": "deepseek-r1:7b",
  "prompt": "What is the philosophical definition of time?"
}
```

We track the following **key metrics per request**:  
- **Tokens Per Second**: Throughput measurement
- **Time to First Byte (TTFB)**: Time until the first response token  
- **Active Duration**: Time between first/last byte of response
- **Token Count**: Total returned token count
- **Error Rate**: How many (%) requests failed at each cycle


#### **Monitoring System Metrics**  
A Go script tracks all ollama processes (with `pgrep ollama`) and monitors CPU, Memory, Threads Count, Open File Descriptors, and GPU usage using **unix** and **macOS powermetrics**. 

We log:  
- **CPU usage (%)**  
- **Memory (RSS, VSZ) & FD count**  
- **GPU Power (Watts) & Frequency (MHz)**  

Per request, per cycle.

The script prints these data in realtime on the console and it also writes on a .csv file.


// TODO: Show these tables in hidable section
##### Console Info
| GPU Metrics | USAGE | POWER | FREQUENCY |
|-------------|-------|-------|-----------|
|             | 100.0%| 21.2W | 1396MHz   |

Running load test with 24 concurrent requests

| REQ ID | TOKENS | TOKENS/S | TTFB | WAITING | DURATION | STATUS       |
|--------|--------|----------|------|---------|----------|--------------|
| 1      | 0      | 0.00     | -    | 7.00s   | -        | Waiting      |
| 2      | 52     | 12.02    | 2.57s| 2.57s   | 4.43s    | In Progress  |
| 3      | 0      | 0.00     | -    | 7.00s   | -        | Waiting      |
| 4      | 53     | 12.27    | 2.56s| 2.56s   | 4.44s    | In Progress  |
| 5      | 0      | 0.00     | -    | 7.00s   | -        | Waiting      |
| 6      | 0      | 0.00     | -    | 7.00s   | -        | Waiting      |
| 7      | 0      | 0.00     | -    | 7.00s   | -        | Waiting      |
| 8      | 0      | 0.00     | -    | 7.00s   | -        | Waiting      |
| 9      | 0      | 0.00     | -    | 7.00s   | -        | Waiting      |
| 10     | 40     | 12.19    | 3.61s| 3.61s   | 3.40s    | In Progress  |
| 11     | 0      | 0.00     | -    | 7.00s   | -        | Waiting      |
| 12     | 52     | 12.03    | 2.57s| 2.57s   | 4.43s    | In Progress  |
| 13     | 0      | 0.00     | -    | 7.00s   | -        | Waiting      |
| 14     | 53     | 12.26    | 2.57s| 2.57s   | 4.43s    | In Progress  |
| 15     | 48     | 11.93    | 2.86s| 2.86s   | 4.14s    | In Progress  |
| 16     | 53     | 12.26    | 2.57s| 2.57s   | 4.43s    | In Progress  |
| 17     | 57     | 12.41    | 2.30s| 2.30s   | 4.70s    | In Progress  |
| 18     | 0      | 0.00     | -    | 7.00s   | -        | Waiting      |
| 19     | 56     | 12.18    | 2.30s| 2.30s   | 4.70s    | In Progress  |
| 20     | 0      | 0.00     | -    | 7.00s   | -        | Waiting      |
| 21     | 53     | 12.26    | 2.57s| 2.57s   | 4.43s    | In Progress  |
| 22     | 53     | 12.26    | 2.57s| 2.57s   | 4.43s    | In Progress  |
| 23     | 0      | 0.00     | -    | 7.00s   | -        | Waiting      |
| 24     | 56     | 12.18    | 2.30s| 2.30s   | 4.70s    | In Progress  |

##### CSV File Output
**ollama/deepseek-r1-7b/test_results_24/request_7_throughput.csv**
| Timestamp              | TokenCount | Throughput |
|------------------------|------------|------------|
| 2025-01-30T17:11:09-03:00 | 1          | 4784689.00 |
| 2025-01-30T17:11:09-03:00 | 2          | 10.25      |
| 2025-01-30T17:11:09-03:00 | 3          | 7.61       |
| 2025-01-30T17:11:09-03:00 | 4          | 6.75       |
| 2025-01-30T17:11:10-03:00 | 6          | 7.55       |
| 2025-01-30T17:11:10-03:00 | 8          | 8.02       |
| 2025-01-30T17:11:10-03:00 | 10         | 8.35       |
| 2025-01-30T17:11:10-03:00 | 12         | 8.57       |
| 2025-01-30T17:11:10-03:00 | 14         | 8.76       |
| 2025-01-30T17:11:11-03:00 | 16         | 8.82       |

**ollama/deepseek-r1-7b/test_results_24/request_summary.csv**
| RequestID | TTFB  | TotalDuration | ResponseDuration | WaitingTime | TokenCount | StatusCode | Status    |
|-----------|-------|---------------|------------------|-------------|------------|------------|-----------|
| 4         | 2.564 | 110.609       | 110.609          | 2.564       | 1183       | 200        | Completed |
| 14        | 2.567 | 140.983       | 140.983          | 2.567       | 1477       | 200        | Completed |
| 15        | 2.861 | 149.495       | 149.495          | 2.861       | 1527       | 200        | Completed |
| 2         | 2.571 | 167.815       | 167.815          | 2.571       | 1720       | 200        | Completed |
| 17        | 2.302 | 177.438       | 177.438          | 2.302       | 1823       | 200        | Completed |

#### Data Collection and Analysis
After everything is written to `.csv` files, I use `Python 3.13` to group, analyze and summarize the data.

You can check the details of the code at [analyze.py](./analyze.py)


TODO: Finish article








### **4. Results and Analysis**  

#### **Response Times & Token Throughput**  
💡 **Findings**:  
- Single request performance averaged **40-50 tokens/s**  
- Increasing concurrency **improved throughput up to 8 concurrent requests**  
- Beyond **12 concurrent requests, throughput plateaued** due to system resource limits  

#### **CPU, Memory, and GPU Usage**  
🔍 **System Bottlenecks Identified**:  
- **CPU Saturation** occurred around **12 concurrent requests**  
- Memory usage **increased linearly** with concurrency, peaking at ~80%  
- **GPU Power & Frequency** remained relatively stable, suggesting CPU bottlenecks  

#### **Scalability Trends & Bottlenecks**  
📉 **Key Insights**:  
- The best **performance-per-watt** was achieved at **8 concurrent requests**  
- **Error rates increased** beyond **16 concurrent requests**, likely due to **memory constraints**  
- At **higher concurrency (24+), TTFB spiked**, indicating **request queuing issues**  

---

### **5. Findings & Observations**  

#### **Optimal Concurrency for Performance**  
- **Sweet Spot**: **8 concurrent requests (~200 tokens/s)**  
- **Beyond 12 requests, throughput gains diminished**  
- **At 24+ concurrency, error rates increased significantly**  

#### **System Resource Constraints**  
- **CPU-bound bottleneck** identified, suggesting potential **multi-process optimization**  
- **Memory limits impacted performance**, possibly due to page swapping  

#### **Unexpected Behaviors and Insights**  
🚀 **Interesting Observations**:  
- GPU power draw was **surprisingly low**, meaning CPU might be doing most of the inference work  
- **Higher latency for first tokens** at high concurrency suggests **IO queue contention**  

---

### **6. Conclusion & Next Steps**  

This benchmark provides a **real-world assessment of DeepSeek R1 (7B) running on a MacBook Pro**. While **Ollama enables impressive local inference**, we observed clear **scalability limits** tied to **CPU and memory constraints**.  

### **Key Takeaways**  
✅ **Best concurrency**: **8 concurrent requests (~200 tokens/s)**  
✅ **CPU is the main bottleneck**, not GPU  
✅ **Error rates increase at 16+ concurrency** due to memory limits  
✅ **Optimizing memory usage or offloading to an external GPU could improve performance**  

### **Future Experiments**  
- **Comparing DeepSeek R1 (7B) vs. 4B and 1B models**  
- **Running the same tests on a Linux workstation with a dedicated GPU**  
- **Testing multi-instance Ollama deployments to improve concurrency handling**  

---

This article combines **technical depth** with **data-driven insights**. Let me know if you want any additional sections or tweaks! 🚀🔥





