package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"strconv"
	"time"
)

const defaultTokensPerSecond = 50.0

func main() {
	tokensPerSecond := defaultTokensPerSecond

	if len(os.Args) > 1 {
		parsedValue, err := strconv.ParseFloat(os.Args[1], 64)
		if err != nil {
			fmt.Printf("Invalid throughput value: %v\n", err)
			os.Exit(1)
		}
		tokensPerSecond = parsedValue
	}

	if tokensPerSecond <= 0 {
		fmt.Println("Throughput must be greater than 0")
		os.Exit(1)
	}

	content, err := ioutil.ReadFile("what-is-the-philosophical-definition-of-time.md")
	if err != nil {
		fmt.Printf("Error reading file: %v\n", err)
		os.Exit(1)
	}

	text := string(content)
	textLength := len(text)
	position := 0

	interval := time.Duration(float64(time.Second) / tokensPerSecond)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for range ticker.C {
		if position >= textLength {
			position = 0
		}

		fmt.Print(string(text[position]))
		position++
	}
} 