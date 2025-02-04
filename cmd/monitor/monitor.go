package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
)

type ProcessMetrics struct {
	Timestamp   time.Time
	PID         int
	ThreadCount int
	FDCount     int
	CPUUsage    float64
	MemUsage    float64
	RSS         int64
	VSZ         int64
}

func monitorProcess(pid int, outputDir string, done chan bool) {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		fmt.Printf("Error creating output directory: %v\n", err)
		return
	}

	csvPath := filepath.Join(outputDir, fmt.Sprintf("process_%d_metrics.csv", pid))
	file, err := os.OpenFile(csvPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		fmt.Printf("Error creating CSV file: %v\n", err)
		return
	}
	
	writer := csv.NewWriter(file)
	writer.Write([]string{
		"Timestamp", "PID", "CPU%", "MEM%", "Threads", "FDs", "VSZ", "RSS",
	})
	defer func() {
		writer.Flush()
		file.Close()
	}()

	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-done:
			return
		case <-ticker.C:
			// Clear the previous line
			fmt.Print("\033[2K\r")
			
			cpuCmd := exec.Command("ps", "-p", strconv.Itoa(pid), "-o", "%cpu,%mem,rss,vsz")
			cpuOutput, err := cpuCmd.Output()
			if err != nil {
				fmt.Printf("Error getting CPU metrics: %v\n", err)
				continue
			}
			
			threadCmd := exec.Command("ps", "M", strconv.Itoa(pid))
			threadOutput, err := threadCmd.Output()
			if err != nil {
				fmt.Printf("Error getting thread count: %v\n", err)
				continue
			}
			threadCount := len(strings.Split(string(threadOutput), "\n")) - 2
			
			fdCmd := exec.Command("lsof", "-p", strconv.Itoa(pid))
			fdOutput, err := fdCmd.Output()
			if err != nil {
				fmt.Printf("Error getting FD count: %v\n", err)
				continue
			}
			fdCount := len(strings.Split(string(fdOutput), "\n")) - 1

			lines := strings.Split(strings.TrimSpace(string(cpuOutput)), "\n")
			if len(lines) >= 2 {
				metrics := strings.Fields(strings.TrimSpace(lines[1]))
				if len(metrics) >= 4 {
					cpu, _ := strconv.ParseFloat(metrics[0], 64)
					mem, _ := strconv.ParseFloat(metrics[1], 64)
					rss, _ := strconv.ParseInt(metrics[2], 10, 64)
					vsz, _ := strconv.ParseInt(metrics[3], 10, 64)

					writer.Write([]string{
						time.Now().Format(time.RFC3339),
						strconv.Itoa(pid),
						fmt.Sprintf("%.2f", cpu),
						fmt.Sprintf("%.2f", mem),
						strconv.Itoa(threadCount),
						strconv.Itoa(fdCount),
						strconv.FormatInt(vsz, 10),
						strconv.FormatInt(rss, 10),
					})
					writer.Flush()

					// Update status with fixed precision for GPU memory
					fmt.Printf("\rPID: %d | CPU: %.1f%% | MEM: %.1f%% | Threads: %d | FDs: %d | RSS: %dMB", 
						pid, cpu, mem, threadCount, fdCount, rss/1024)
				}
			}
		}
	}
}

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintf(os.Stderr, "Usage: monitor <pid> <output_directory>\n")
		os.Exit(1)
	}

	pid, err := strconv.Atoi(os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "Invalid PID: %v\n", err)
		os.Exit(1)
	}

	outputDir := os.Args[2]
	fmt.Printf("Starting monitor for PID %d\nOutput directory: %s\n", pid, outputDir)
	fmt.Println("Press Ctrl+C to stop monitoring")

	done := make(chan bool)
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go monitorProcess(pid, outputDir, done)

	<-sigChan
	done <- true
	fmt.Print("\033[2K\r") // Clear the last status line
	fmt.Println("\nMonitoring stopped")
	
	cmd := exec.Command("osascript", "-e", `tell application "Terminal" to close (every window whose name contains "monitor.go")`)
	if err := cmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error closing terminal: %v\n", err)
		os.Exit(1)
	}
} 