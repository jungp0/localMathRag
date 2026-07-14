package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"localmathrag/mtef-parser/eqn"
)

func main() {
	filePath := flag.String("f", "", "Equation Editor OLE object path; stdin is used when omitted")
	flag.Parse()

	var data []byte
	var err error
	if *filePath != "" {
		data, err = os.ReadFile(*filePath)
	} else {
		data, err = io.ReadAll(os.Stdin)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	mtef, err := eqn.Open(bytes.NewReader(data))
	if err != nil || mtef == nil {
		if err == nil {
			err = fmt.Errorf("Equation Native stream was not found")
		}
		fmt.Fprintln(os.Stderr, err)
		os.Exit(3)
	}

	latex := strings.TrimSpace(mtef.Translate())
	if latex == "" {
		fmt.Fprintln(os.Stderr, "MTEF stream did not produce LaTeX")
		os.Exit(4)
	}
	fmt.Print(latex)
}
