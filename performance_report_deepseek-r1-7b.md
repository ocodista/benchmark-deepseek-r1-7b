# DeepSeek R1 7B Performance Analysis

## GPU Energy Analysis

- **Total Test Duration:** 68h 21m 54s
- **Average Power:** 20.4W
- **Total Energy:** 278.34Wh (0.2783kWh)
- **Energy Cost:** R$0.1948

### GPU Metrics by Parallel Request Level

| Parallel Requests | Avg Power (W) | Energy (Wh) | Duration | Avg GPU Usage % |
|-------------------|---------------|-------------|-----------|----------------|
|                 1 |        16.6 |       0.13 |     27.0s |          100.0 |
|                 2 |        20.6 |      49.02 |   8564.0s |          100.0 |
|                 4 |        20.6 |      28.96 |   5064.0s |          100.0 |
|                 8 |        20.5 |      28.24 |   4952.0s |          100.0 |
|                16 |        22.3 |       1.74 |    280.0s |          100.0 |
|                19 |        20.3 |      26.91 |   4777.0s |          100.0 |
|                32 |        20.4 |      42.68 |   7544.0s |          100.0 |
|                38 |        20.9 |       3.41 |    587.0s |          100.0 |
|                57 |        20.2 |      21.71 |   3865.0s |          100.0 |
|                64 |        20.9 |       6.47 |   1114.0s |          100.0 |
|                76 |        20.4 |       7.52 |   1329.0s |          100.0 |
|                95 |        20.4 |       9.00 |   1587.0s |          100.0 |
|               128 |        20.3 |      32.68 |   5794.0s |          100.0 |
|               256 |        20.2 |      19.88 |   3536.0s |          100.0 |

## Performance

![Performance Metrics](./performance_metrics.png)

This chart shows the throughput scaling and test duration across different parallel request levels.

## Token Generation Distribution

![Token Distribution](./token_distribution.png)

This visualization shows the token generation speed over time for different concurrency levels.

## Process Resource Usage

![Process Metrics](./process_metrics.png)

This chart shows CPU, Memory, and Thread usage of the Ollama processes during the test.

## Detailed Metrics

| Concurrency | Avg Tokens/s | P95 Tokens/s | P99 Tokens/s | Avg Waiting Time | Error Rate | Duration | P99 Duration | Total Tokens |
|------------|--------------|--------------|--------------|-----------------|------------|-----------|--------------|--------------|
|        1.0 |        55.1 |        55.1 |        55.1 |           0.94 |        0.0 | 00:39.33 |    00:39.33 |       5,507 |
|        2.0 |        30.7 |        30.9 |        30.9 |           0.36 |        0.0 | 01:12.09 |    01:12.27 |       4,422 |
|        4.0 |        21.5 |        22.4 |        22.6 |           1.54 |        0.0 | 01:34.31 |    01:47.17 |      14,956 |
|        8.0 |        12.7 |        13.2 |        13.3 |           0.47 |        0.0 | 02:32.12 |    02:54.62 |      30,560 |
|       16.0 |         9.8 |        10.1 |        10.3 |           0.64 |        0.0 | 03:26.81 |    04:41.07 |      32,418 |
|       19.0 |         9.0 |         9.4 |         9.5 |           0.62 |        0.0 | 03:43.55 |    05:23.83 |      77,486 |
|       32.0 |         8.0 |         9.3 |         9.3 |          84.05 |        0.0 |  04:8.64 |    06:15.66 |      62,452 |
|       38.0 |         7.7 |         9.3 |         9.6 |         110.13 |        0.0 | 04:17.54 |    05:45.43 |      73,385 |
|       57.0 |         7.2 |         9.2 |         9.4 |         250.38 |        0.0 | 04:30.52 |     06:5.81 |     109,573 |
|       64.0 |         7.1 |         9.0 |         9.3 |         321.96 |        0.0 | 04:39.51 |    07:30.13 |     124,372 |
|       76.0 |         7.0 |         8.9 |         9.3 |         400.85 |        0.0 | 04:46.76 |    07:40.30 |     150,573 |
|       95.0 |         6.6 |         8.9 |         9.2 |         576.43 |        0.0 | 04:59.59 |    08:40.30 |     185,522 |
|      128.0 |         6.5 |         8.9 |         9.2 |         858.27 |        0.0 |  05:6.60 |    07:28.05 |     251,551 |
|      256.0 |         6.3 |         8.8 |         9.4 |        1534.79 |        0.0 | 04:12.89 |    07:52.97 |     404,069 |

## Key Findings

- **Optimal Concurrency:** 1.0 concurrent requests
- **Peak Performance:** 55.1 tokens/s average
- **Scaling Factor:** 1.0x speedup from single request

## Process Resource Details

| Process ID | Avg CPU % | Max CPU % | Avg Memory % | Max Memory % | Avg Threads | Max RSS (MB) |
|------------|-----------|-----------|--------------|--------------|-------------|--------------|
|      40163 |      0.2 |     22.8 |         0.6 |         0.7 |       46.7 | 114.1 |
|      42567 |      5.5 |     14.3 |        14.4 |        14.6 |       17.6 | 2397.7 |
|      94405 |      5.7 |     14.9 |        14.4 |        14.6 |       16.9 | 2385.0 |
|      68068 |      0.2 |      4.6 |         0.6 |         0.6 |       37.4 | 103.9 |
|       6473 |      2.1 |     48.0 |         0.1 |         0.5 |       19.0 | 75.3 |
|      17115 |      4.2 |      6.0 |         0.2 |         0.2 |       14.4 | 36.9 |
|      81655 |      4.9 |      8.5 |         0.3 |         0.3 |       16.4 | 55.6 |

## GPU Resource Usage

![GPU Metrics](./gpu_metrics.png)

This chart shows GPU usage, power consumption, and frequency across different concurrency levels.


## GPU Metrics Details

| Concurrency | Avg Usage % | Max Usage % | Avg Power (W) | Max Power (W) | Avg Freq (MHz) | Max Freq (MHz) |
|------------|-------------|-------------|---------------|---------------|----------------|----------------|
|          1 |      100.0 |      100.0 |         16.6 |         19.7 |          1299 |          1391 |
|          2 |      100.0 |      100.0 |         20.6 |         31.8 |          1385 |          1397 |
|          4 |      100.0 |      100.0 |         20.6 |         31.8 |          1382 |          1397 |
|          8 |      100.0 |      100.0 |         20.5 |         31.9 |          1383 |          1397 |
|         16 |      100.0 |      100.0 |         22.3 |         29.3 |          1392 |          1397 |
|         19 |      100.0 |      100.0 |         20.3 |         29.4 |          1385 |          1397 |
|         32 |      100.0 |      100.0 |         20.4 |         29.3 |          1391 |          1397 |
|         38 |      100.0 |      100.0 |         20.9 |         27.2 |          1397 |          1397 |
|         57 |      100.0 |      100.0 |         20.2 |         28.9 |          1388 |          1397 |
|         64 |      100.0 |      100.0 |         20.9 |         28.1 |          1397 |          1397 |
|         76 |      100.0 |      100.0 |         20.4 |         27.3 |          1396 |          1397 |
|         95 |      100.0 |      100.0 |         20.4 |         26.7 |          1396 |          1397 |
|        128 |      100.0 |      100.0 |         20.3 |         29.1 |          1393 |          1397 |
|        256 |      100.0 |      100.0 |         20.2 |         29.1 |          1394 |          1397 |