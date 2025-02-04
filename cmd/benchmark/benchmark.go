package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type RequestPayload struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type ResponseChunk struct {
	Model    string `json:"model"`
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

type RequestMetrics struct {
	ID              int
	StartTime       time.Time
	FirstTokenTime  time.Time
	EndTime         time.Time
	TTFB            float64
	TotalDuration   float64
	ResponseDuration float64
	TokenCount      int
	WaitingTime     float64
	StatusCode      int
	Measurements    []Measurement
	TokenOutput     []string
}

type Measurement struct {
	Timestamp  time.Time
	TokenCount int
	Throughput float64
}

type RequestSummary struct {
	ID           int
	AvgThroughput float64
	P95Throughput float64
	P99Throughput float64
}

type GPUMetrics struct {
	Timestamp time.Time
	Usage     float64
	Power     float64
	Frequency float64
}

type GPUStatus struct {
	Usage     float64
	Power     float64
	Frequency float64
}

type DurationMetrics struct {
	MaxDuration    float64
	MinDuration    float64
	AvgDuration    float64
	P99Duration    float64
	Concurrency    int
	Timestamp      time.Time
}

func getOllamaPIDs() []int {
	cmd := exec.Command("pgrep", "ollama")
	output, err := cmd.Output()
	if err != nil {
		fmt.Printf("Error getting ollama PIDs: %v\n", err)
		return nil
	}

	var pids []int
	for _, pidStr := range strings.Split(strings.TrimSpace(string(output)), "\n") {
		if pidStr != "" {
			pid, _ := strconv.Atoi(pidStr)
			pids = append(pids, pid)
		}
	}
	return pids
}

func launchMonitor(pid int, testDir string) error {
	// Get current working directory
	pwd, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("failed to get working directory: %v", err)
	}

	// Create full path for output
	outputPath := filepath.Join(pwd, testDir)

	script := fmt.Sprintf(`tell application "Terminal"
		do script "cd '%s' && go run ./cmd/monitor/monitor.go %d '%s'"
	end tell`, pwd, pid, outputPath)
	
	cmd := exec.Command("osascript", "-e", script)
	return cmd.Run()
}

func performRequest(id int, metrics chan<- RequestMetrics, activeRequests *sync.Map, model string) {
	requestMetrics := RequestMetrics{
		ID:        id,
		StartTime: time.Now(),
	}
	activeRequests.Store(id, &requestMetrics)

	payload := RequestPayload{
		Model:  model,
		Prompt: "What is the philosophical definition of time?",
	}

	jsonData, _ := json.Marshal(payload)
	req, _ := http.NewRequest("POST", "http://localhost:11434/api/generate", strings.NewReader(string(jsonData)))
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		requestMetrics.StatusCode = http.StatusInternalServerError
		requestMetrics.EndTime = time.Now()
		metrics <- requestMetrics
		activeRequests.Store(id, &requestMetrics)
		return
	}
	defer resp.Body.Close()

	requestMetrics.StatusCode = resp.StatusCode
	if resp.StatusCode != http.StatusOK {
		requestMetrics.EndTime = time.Now()
		metrics <- requestMetrics
		activeRequests.Store(id, &requestMetrics)
		return
	}

	reader := bufio.NewReader(resp.Body)
	firstToken := true

	for {
		line, err := reader.ReadBytes('\n')
		if err == io.EOF {
			break
		}
		if err != nil {
			fmt.Printf("Error reading response: %v\n", err)
			break
		}

		var chunk ResponseChunk
		if err := json.Unmarshal(line, &chunk); err != nil {
			fmt.Printf("Error unmarshalling response: %v\n", err)
			continue
		}

		if len(chunk.Response) > 0 {
			requestMetrics.TokenOutput = append(requestMetrics.TokenOutput, chunk.Response)
		}

		if firstToken && len(chunk.Response) > 0 {
			requestMetrics.FirstTokenTime = time.Now()
			requestMetrics.TTFB = time.Since(requestMetrics.StartTime).Seconds()
			requestMetrics.WaitingTime = requestMetrics.TTFB
			firstToken = false
		}

		requestMetrics.TokenCount += len(strings.Split(chunk.Response, " "))
		now := time.Now()
		
		if !requestMetrics.FirstTokenTime.IsZero() {
			elapsed := now.Sub(requestMetrics.FirstTokenTime).Seconds()
			if elapsed > 0 {
				throughput := float64(requestMetrics.TokenCount) / elapsed
				measurement := Measurement{
					Timestamp:  now,
					TokenCount: requestMetrics.TokenCount,
					Throughput: throughput,
				}
				requestMetrics.Measurements = append(requestMetrics.Measurements, measurement)
			}
		}

		activeRequests.Store(id, &requestMetrics)

		if chunk.Done {
			requestMetrics.EndTime = time.Now()
			if !requestMetrics.FirstTokenTime.IsZero() {
				requestMetrics.TotalDuration = requestMetrics.EndTime.Sub(requestMetrics.FirstTokenTime).Seconds()
				requestMetrics.ResponseDuration = requestMetrics.TotalDuration
			}
			metrics <- requestMetrics
			activeRequests.Store(id, &requestMetrics)
			break
		}
	}
}

func saveRequestMetrics(metrics RequestMetrics, testDir string) error {
	status := "Waiting"
	if metrics.TokenCount > 0 {
		if !metrics.EndTime.IsZero() {
			if metrics.StatusCode != http.StatusOK {
				status = fmt.Sprintf("Error %d", metrics.StatusCode)
			} else {
				status = "Completed"
			}
		} else {
			status = "In Progress"
		}
	}

	// Ensure directory exists
	if err := os.MkdirAll(testDir, 0755); err != nil {
		return fmt.Errorf("failed to create directory %s: %v", testDir, err)
	}

	// Save summary metrics
	summaryPath := filepath.Join(testDir, "request_summary.csv")
	record := []string{
		fmt.Sprintf("%d", metrics.ID),
		fmt.Sprintf("%.3f", metrics.TTFB),
		fmt.Sprintf("%.3f", metrics.TotalDuration),
		fmt.Sprintf("%.3f", metrics.ResponseDuration),
		fmt.Sprintf("%.3f", metrics.TTFB),
		fmt.Sprintf("%d", metrics.TokenCount),
		fmt.Sprintf("%d", metrics.StatusCode),
		status,
	}

	// Create file if it doesn't exist
	var file *os.File
	
	if _, err := os.Stat(summaryPath); os.IsNotExist(err) {
		file, err = os.Create(summaryPath)
		if err != nil {
			return fmt.Errorf("failed to create file %s: %v", summaryPath, err)
		}
		writer := csv.NewWriter(file)
		if err := writer.Write([]string{
			"RequestID",
			"TTFB",
			"TotalDuration",
			"ResponseDuration",
			"WaitingTime",
			"TokenCount",
			"StatusCode",
			"Status",
		}); err != nil {
			file.Close()
			return fmt.Errorf("failed to write header: %v", err)
		}
		writer.Flush()
	} else {
		file, err = os.OpenFile(summaryPath, os.O_APPEND|os.O_WRONLY, 0644)
		if err != nil {
			return fmt.Errorf("failed to open file %s: %v", summaryPath, err)
		}
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	if err := writer.Write(record); err != nil {
		return fmt.Errorf("failed to write record: %v", err)
	}

	return nil
}

func isValidHeader(header []string) bool {
	expectedHeader := []string{
		"RequestID",
		"TTFB",
		"TotalDuration",
		"ResponseDuration",
		"WaitingTime",
		"TokenCount",
		"StatusCode",
		"Status",
	}

	if len(header) != len(expectedHeader) {
		return false
	}

	for i, field := range header {
		if field != expectedHeader[i] {
			return false
		}
	}

	return true
}

func appendToCSV(filepath string, record []string, header string) error {
	var file *os.File
	
	if _, err := os.Stat(filepath); os.IsNotExist(err) {
		file, err = os.Create(filepath)
		if err != nil {
			return err
		}
		writer := csv.NewWriter(file)
		writer.Write(strings.Split(header, ","))
		writer.Flush()
	} else {
		file, err = os.OpenFile(filepath, os.O_APPEND|os.O_WRONLY, 0644)
		if err != nil {
			return err
		}
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()
	return writer.Write(record)
}

func writeCSV(filepath string, measurements []Measurement) error {
	file, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	writer.Write([]string{"Timestamp", "TokenCount", "Throughput"})
	
	for _, m := range measurements {
		writer.Write([]string{
			m.Timestamp.Format(time.RFC3339),
			fmt.Sprintf("%d", m.TokenCount),
			fmt.Sprintf("%.2f", m.Throughput),
		})
	}

	return nil
}

func printStatus(concurrencyLevel int, activeRequests *sync.Map, gpuStatus *GPUStatus, noCharts bool, logSections []string) {
	clearScreen()

	for _, section := range logSections {
		switch section {
		case "gpu":
			printGPUMetrics(gpuStatus)
		case "table":
			printRequestsTable(concurrencyLevel, activeRequests)
		case "chart":
			if !noCharts {
				printMetricsChart(activeRequests)
			}
		case "response":
			printResponseOutput(activeRequests)
		}
	}
}

func clearScreen() {
	fmt.Print("\033[2J\033[H")
}

func printGPUMetrics(gpuStatus *GPUStatus) {
	fmt.Println("GPU Metrics")
	fmt.Printf("%12s %12s %12s\n", "USAGE", "POWER", "FREQUENCY")
	fmt.Println(strings.Repeat("-", 40))
	fmt.Printf("%11.1f%% %11.1fW %11.0fMHz\n\n",
		gpuStatus.Usage, gpuStatus.Power, gpuStatus.Frequency)
}

func printRequestsTable(concurrencyLevel int, activeRequests *sync.Map) {
	fmt.Printf("Running load test with %d concurrent requests\n\n", concurrencyLevel)
	printTableHeader()
	printTableRows(concurrencyLevel, activeRequests)
}

func printTableHeader() {
	fmt.Printf("%8s %12s %12s %10s %10s %15s %15s\n",
		"REQ ID", "TOKENS", "TOKENS/S", "TTFB", "WAITING", "DURATION", "STATUS")
	fmt.Println(strings.Repeat("-", 90))
}

func printTableRows(concurrencyLevel int, activeRequests *sync.Map) {
	for i := 1; i <= concurrencyLevel; i++ {
		if metrics, ok := activeRequests.Load(i); ok {
			printActiveRequest(i, metrics.(*RequestMetrics))
		} else {
			printWaitingRequest(i)
		}
	}
}

func printActiveRequest(id int, m *RequestMetrics) {
	ttfb := formatTTFB(m)
	duration := formatDuration(m)
	waiting := formatWaiting(m)
	status := getRequestStatus(m)
	throughput := getLatestThroughput(m)

	fmt.Printf("%8d %12d %12.2f %10s %10s %15s %15s\n",
		id, m.TokenCount, throughput, ttfb, waiting, duration, status)
}

func printWaitingRequest(id int) {
	fmt.Printf("%8d %12s %12s %10s %10s %15s %15s\n",
		id, "-", "-", "-", "-", "-", "Waiting")
}

func formatTTFB(m *RequestMetrics) string {
	if !m.FirstTokenTime.IsZero() {
		return fmt.Sprintf("%.2fs", m.TTFB)
	}
	return "-"
}

func formatDuration(m *RequestMetrics) string {
	if m.FirstTokenTime.IsZero() {
		return "-"
	}
	if !m.EndTime.IsZero() {
		return fmt.Sprintf("%.2fs", m.TotalDuration)
	}
	return fmt.Sprintf("%.2fs", time.Since(m.FirstTokenTime).Seconds())
}

func formatWaiting(m *RequestMetrics) string {
	if m.TokenCount == 0 {
		return fmt.Sprintf("%.2fs", time.Since(m.StartTime).Seconds())
	}
	return fmt.Sprintf("%.2fs", m.TTFB)
}

func getLatestThroughput(m *RequestMetrics) float64 {
	if len(m.Measurements) > 0 {
		return m.Measurements[len(m.Measurements)-1].Throughput
	}
	return 0
}

func printMetricsChart(activeRequests *sync.Map) {
	fmt.Println("\nMetrics Chart (last 60s)")
	fmt.Println(strings.Repeat("-", 90))

	chartData := collectChartData(activeRequests)
	maxValues := findMaxValues(chartData)
	printCharts(chartData, maxValues)
}

func collectChartData(activeRequests *sync.Map) [][]float64 {
	timeRange := 60
	chartData := make([][]float64, timeRange)
	now := time.Now()

	for i := range chartData {
		chartData[i] = make([]float64, 4)
	}

	activeRequests.Range(func(key, value interface{}) bool {
		m := value.(*RequestMetrics)
		if len(m.Measurements) > 0 {
			for _, measurement := range m.Measurements {
				secondsAgo := int(now.Sub(measurement.Timestamp).Seconds())
				if secondsAgo < timeRange {
					idx := timeRange - secondsAgo - 1
					if idx >= 0 && idx < timeRange {
						chartData[idx][0] = float64(measurement.TokenCount)
						chartData[idx][1] = measurement.Throughput
					}
				}
			}
		}
		return true
	})

	return chartData
}

func findMaxValues(chartData [][]float64) []float64 {
	maxValues := make([]float64, 4)
	for _, row := range chartData {
		for i, val := range row {
			if val > maxValues[i] {
				maxValues[i] = val
			}
		}
	}
	return maxValues
}

func printCharts(chartData [][]float64, maxValues []float64) {
	chartHeight := 10
	timeRange := len(chartData)
	labels := []string{"Tokens", "Tokens/s"}

	for i, label := range labels {
		if maxValues[i] == 0 {
			continue
		}

		fmt.Printf("\n%s (max: %.0f)\n", label, maxValues[i])
		printChart(chartData, maxValues[i], i, chartHeight, timeRange)
	}
}

func printChart(chartData [][]float64, maxValue float64, dataIndex, height, timeRange int) {
	for h := height - 1; h >= 0; h-- {
		threshold := float64(h) * maxValue / float64(height-1)
		fmt.Printf("%6.0f │", threshold)
		
		for t := 0; t < timeRange; t++ {
			value := chartData[t][dataIndex]
			if value >= threshold {
				fmt.Print("█")
			} else {
				fmt.Print(" ")
			}
		}
		fmt.Println()
	}
	
	fmt.Print("      └" + strings.Repeat("─", timeRange) + "\n")
	fmt.Print("      " + strings.Repeat(" ", timeRange/2-2) + "Time (s)\n")
}

func printResponseOutput(activeRequests *sync.Map) {
	fmt.Println("\nResponse Output")
	fmt.Println(strings.Repeat("-", 90))

	if metrics, ok := activeRequests.Load(1); ok {
		m := metrics.(*RequestMetrics)
		if len(m.TokenOutput) > 0 {
			fmt.Printf("Request #1: %s\n", getRequestStatus(m))
			fmt.Printf("Prompt: What is the philosophical definition of time?\n\n")
			fmt.Println(strings.Join(m.TokenOutput, ""))
		} else {
			fmt.Println("Waiting for response...")
		}
	} else {
		fmt.Println("Request #1 not started yet")
	}
	fmt.Println(strings.Repeat("-", 90))
}

func getRequestStatus(m *RequestMetrics) string {
	if m.TokenCount == 0 {
		return "Waiting"
	}
	if !m.EndTime.IsZero() {
		if m.StatusCode != http.StatusOK {
			return fmt.Sprintf("Error %d", m.StatusCode)
		}
		return "Completed"
	}
	return "In Progress"
}

func monitorGPU(testDir string, done chan bool) error {
	csvPath := filepath.Join(testDir, "gpu_metrics.csv")
	file, err := os.Create(csvPath)
	if err != nil {
		return fmt.Errorf("failed to create GPU metrics file: %v", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	writer.Write([]string{"Timestamp", "Usage%", "Power(W)", "Frequency(MHz)"})

	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-done:
			return nil
		case <-ticker.C:
			metrics, err := getGPUMetrics()
			if err != nil {
				fmt.Printf("Error getting GPU metrics: %v\n", err)
				continue
			}

			writer.Write([]string{
				time.Now().Format(time.RFC3339),
				fmt.Sprintf("%.2f", metrics.Usage),
				fmt.Sprintf("%.2f", metrics.Power),
				fmt.Sprintf("%.2f", metrics.Frequency),
			})
			writer.Flush()
		}
	}
}

func getGPUMetrics() (GPUMetrics, error) {
	cmd := exec.Command("sudo", "powermetrics", "--samplers", "gpu_power", "-i", "1000", "-n", "1")
	output, err := cmd.Output()
	if err != nil {
		return GPUMetrics{}, err
	}

	lines := strings.Split(string(output), "\n")
	var metrics GPUMetrics
	metrics.Timestamp = time.Now()

	for _, line := range lines {
		line = strings.TrimSpace(line)
		
		if strings.Contains(line, "GPU idle residency:") {
			parts := strings.Split(line, ": ")
			if len(parts) == 2 {
				idleStr := strings.TrimSuffix(parts[1], "%")
				idle, _ := strconv.ParseFloat(idleStr, 64)
				metrics.Usage = 100 - idle
			}
		}
		
		if strings.Contains(line, "GPU HW active frequency:") {
			parts := strings.Split(line, ": ")
			if len(parts) == 2 {
				freqStr := strings.TrimSuffix(parts[1], " MHz")
				metrics.Frequency, _ = strconv.ParseFloat(freqStr, 64)
			}
		}
		
		if strings.Contains(line, "GPU Power:") {
			parts := strings.Split(line, ": ")
			if len(parts) == 2 {
				powerStr := strings.TrimSuffix(parts[1], " mW")
				powerMw, _ := strconv.ParseFloat(powerStr, 64)
				metrics.Power = powerMw / 1000
			}
		}
	}

	return metrics, nil
}

func calculateDurationMetrics(metrics []RequestMetrics, concurrency int) DurationMetrics {
	if len(metrics) == 0 {
		return DurationMetrics{
			Concurrency: concurrency,
			Timestamp:   time.Now(),
		}
	}

	durations := make([]float64, 0, len(metrics))
	var sum float64

	for _, m := range metrics {
		duration := m.TotalDuration
		durations = append(durations, duration)
		sum += duration
	}

	// Sort for percentile calculation
	sort.Float64s(durations)

	p99Index := int(float64(len(durations)) * 0.99)
	if p99Index >= len(durations) {
		p99Index = len(durations) - 1
	}

	return DurationMetrics{
		MaxDuration:    durations[len(durations)-1],
		MinDuration:    durations[0],
		AvgDuration:    sum / float64(len(durations)),
		P99Duration:    durations[p99Index],
		Concurrency:    concurrency,
		Timestamp:      time.Now(),
	}
}

func saveDurationMetrics(metrics DurationMetrics, testDir string) error {
	filepath := filepath.Join(testDir, "duration_metrics.csv")
	
	record := []string{
		metrics.Timestamp.Format(time.RFC3339),
		fmt.Sprintf("%d", metrics.Concurrency),
		fmt.Sprintf("%.3f", metrics.MaxDuration),
		fmt.Sprintf("%.3f", metrics.MinDuration),
		fmt.Sprintf("%.3f", metrics.AvgDuration),
		fmt.Sprintf("%.3f", metrics.P99Duration),
	}
	
	return appendToCSV(filepath, record, "Timestamp,Concurrency,MaxDuration,MinDuration,AvgDuration,P99Duration")
}

func runPhase(concurrencyLevel int, baseDir string, noCharts bool, logSections []string, model string) error {
	testDir := filepath.Join(baseDir, fmt.Sprintf("test_results_%d", concurrencyLevel))
	if err := os.MkdirAll(testDir, 0755); err != nil {
		return fmt.Errorf("failed to create test directory: %v", err)
	}

	metrics := make(chan RequestMetrics, concurrencyLevel)
	var wg sync.WaitGroup
	activeRequests := &sync.Map{}

	// Start GPU monitoring
	gpuDone := make(chan bool)
	gpuStatus := &GPUStatus{}
	
	go monitorGPU(testDir, gpuDone)
	go func() {
		for {
			select {
			case <-gpuDone:
				return
			default:
				if metrics, err := getGPUMetrics(); err == nil {
					gpuStatus.Usage = metrics.Usage
					gpuStatus.Power = metrics.Power
					gpuStatus.Frequency = metrics.Frequency
				}
				time.Sleep(time.Second)
			}
		}
	}()

	// Start status printer
	done := make(chan bool)
	go func() {
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				printStatus(concurrencyLevel, activeRequests, gpuStatus, noCharts, logSections)
			}
		}
	}()

	// Start load test
	fmt.Printf("\nStarting load test with %d concurrent requests\n", concurrencyLevel)
	for i := 0; i < concurrencyLevel; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			performRequest(id+1, metrics, activeRequests, model)
		}(i)
	}

	// Wait 1s before starting monitors
	time.Sleep(time.Second)

	// Launch process monitors
	pids := getOllamaPIDs()
	for _, pid := range pids {
		if err := launchMonitor(pid, testDir); err != nil {
			fmt.Printf("Failed to launch monitor for PID %d: %v\n", pid, err)
		}
	}

	// Wait for completion and collect metrics
	go func() {
		wg.Wait()
		close(metrics)
	}()

	// Create slice to store all request metrics
	var allMetrics []RequestMetrics

	// Modify the metrics collection loop
	for metric := range metrics {
		allMetrics = append(allMetrics, metric)
		if err := saveRequestMetrics(metric, testDir); err != nil {
			fmt.Printf("Error saving metrics for request %d: %v\n", metric.ID, err)
		}
	}

	// Calculate and save duration metrics
	durationMetrics := calculateDurationMetrics(allMetrics, concurrencyLevel)
	if err := saveDurationMetrics(durationMetrics, testDir); err != nil {
		fmt.Printf("Error saving duration metrics: %v\n", err)
	}

	// Signal goroutines to stop
	done <- true
	gpuDone <- true

	// Kill all monitor terminals
	cmd := exec.Command("osascript", "-e", `tell application "Terminal"
		set windowList to every window whose name contains "monitor.go"
		repeat with aWindow in windowList
			tell aWindow
				do script "sudo pkill -f monitor.go" in selected tab
				delay 0.5
				close
			end tell
		end repeat
	end tell`)
	if err := cmd.Run(); err != nil {
		fmt.Printf("Error closing monitor terminals: %v\n", err)
	}

	fmt.Printf("\nPhase completed. Waiting 10s before next phase...\n")
	time.Sleep(10 * time.Second)

	return nil
}

func parseConcurrencyLevels(input string) ([]int, error) {
	parts := strings.Split(input, ",")
	levels := make([]int, 0, len(parts))
	
	for _, part := range parts {
		level, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil {
			return nil, fmt.Errorf("invalid concurrency level '%s': %v", part, err)
		}
		if level < 1 {
			return nil, fmt.Errorf("concurrency level must be greater than 0: %d", level)
		}
		levels = append(levels, level)
	}
	
	return levels, nil
}

func modelToDir(model string) string {
	return strings.ReplaceAll(model, ":", "-")
}

func main() {
	noCharts := flag.Bool("no-charts", false, "Disable chart printing")
	logOptions := flag.String("log", "gpu,table,chart,response", "Comma-separated list of sections to log: gpu, table, chart, response")
	model := flag.String("model", "deepseek-r1:7b", "Model to benchmark")

	flag.Parse()

	if len(flag.Args()) != 1 {
		fmt.Println("Usage: benchmark [options] <concurrency_levels>")
		fmt.Println("Options:")
		flag.PrintDefaults()
		os.Exit(1)
	}

	concurrencyLevels, err := parseConcurrencyLevels(flag.Args()[0])
	if err != nil {
		fmt.Printf("Error parsing concurrency levels: %v\n", err)
		os.Exit(1)
	}

	baseDir := filepath.Join("ollama", modelToDir(*model))
	if err := os.MkdirAll(baseDir, 0755); err != nil {
		fmt.Printf("Failed to create base directory: %v\n", err)
		return
	}

	logSections := strings.Split(*logOptions, ",")

	for _, level := range concurrencyLevels {
		if err := runPhase(level, baseDir, *noCharts, logSections, *model); err != nil {
			fmt.Printf("Error running phase with concurrency %d: %v\n", level, err)
			return
		}
	}

	fmt.Println("\nBenchmark completed!")
} 
