package LogisticRegression

import(
    
    "fmt"
    "../utils"
)

type LogisticRegression struct{

    in int
    out int
    miniBatchSize int
    
    W [][]float64
    Bias []float64
    
    Output [][]float64
    Delta [][]float64
    CrossEntropy float64
    AccuracyCount int
    PredictedLabel [][]int
}

func Construct(self *LogisticRegression, in int, out int, miniBatchSize int){

    self.in = in
    self.out = out
    self.miniBatchSize = miniBatchSize

    var randomBoundary float64 = 1.0 / float64(in)
    self.W = make([][]float64, out)
    for i := 0; i < out; i++{
        self.W[i] = make([]float64, in)
    }
    for i := 0; i < out; i++{
        for j := 0; j < in; j++{
            self.W[i][j] = utils.Uniform(-randomBoundary, randomBoundary)
        }
    }

    self.Bias = make([]float64, out)
}

func Confirm(self *LogisticRegression){

    fmt.Println("In")
    fmt.Println(self.in)
    fmt.Println("Label")
    fmt.Println(self.out)
    fmt.Println("miniBatchSize")
    fmt.Println(self.miniBatchSize)
}

func Output(self *LogisticRegression, input [][]float64){

    var output = make([][]float64, self.miniBatchSize)
    for i := 0; i < self.miniBatchSize; i++{
        output[i] = make([]float64, self.out)
    }

    for i := 0; i < self.miniBatchSize; i++{
        for j := 0; j < self.out; j++{
            var out float64 = 0.0

            for k := 0; k < self.in; k++{
                out += self.W[j][k] * input[i][k]
            }
            output[i][j] = out + self.Bias[j]
        }
    }
    self.Output = output
}

func Train(self *LogisticRegression, input [][]float64, actualLabel[][]int, learningRate float64){

    var gradW = make([][]float64, self.out)
    for i := 0; i < self.out; i++{
        gradW[i] = make([]float64, self.in)
    }
    
    var gradBias = make([]float64, self.out)

    var delta = make([][]float64, self.miniBatchSize)
    for i := 0; i < self.miniBatchSize; i++{
        delta[i] = make([]float64, self.out)
    }

    var crossEntropy float64 = 0.0

    Output(self, input)
    for i := 0; i < self.miniBatchSize; i++{
        self.Output[i] = append(utils.SoftMax(self.Output[i]))
        
        fmt.Println("ActualLabel")
        fmt.Println(actualLabel[i])
        fmt.Println("SoftMax")
        fmt.Println(self.Output[i])

        crossEntropy += utils.CrossEntropy(self.Output[i], actualLabel[i])
    }
    
    for i := 0; i < self.miniBatchSize; i++{
        for j := 0; j < self.out; j++{
            delta[i][j] = self.Output[i][j] - float64(actualLabel[i][j])

            gradBias[j] += delta[i][j]
            
            for k := 0; k < self.in; k++{
                gradW[j][k] += delta[i][j] * input[i][k]
            }
        }
    }
    
    for i := 0; i < self.out; i++{
        for j := 0; j < self.in; j++{
            self.W[i][j] -= learningRate * gradW[i][j] / float64(self.miniBatchSize)
        }
        self.Bias[i] -= learningRate * gradBias[i] / float64(self.miniBatchSize)
    }
    self.CrossEntropy = crossEntropy
    self.Delta = delta
}

func Predict(self *LogisticRegression, input [][]float64, actualLabel [][]int){

    var argMax []int
    argMax = make([]int, self.miniBatchSize)

    var predictedLabel = make([][]int, self.miniBatchSize)
    for i := 0; i < self.miniBatchSize; i++{
        predictedLabel[i] = make([]int, self.out)
    }

    var accuracyCount int = 0
    
    Output(self, input)
    for i := 0; i < self.miniBatchSize; i++{
        self.Output[i] = append(utils.SoftMax(self.Output[i]))
    }

    for i := 0; i < self.miniBatchSize; i++{
        var max float64 = 0.0
        for j := 0; j < self.out; j++{
            if self.Output[i][j] > max{
                max = self.Output[i][j]
                argMax[i] = j
            }
        }
    }

    for i := 0; i < self.miniBatchSize; i++{
        for j := 0; j < self.out; j++{
            if argMax[i] == j{
                predictedLabel[i][j] = 1
            }else{
                predictedLabel[i][j] = 0
            }
        }
    }
    
    for i := 0; i < self.miniBatchSize; i++{

        fmt.Println()
        fmt.Println("ActualLabel")
        fmt.Println(actualLabel[i])
        fmt.Println("SoftMax")
        fmt.Println(self.Output[i])
        fmt.Println("PredictedLabel")
        fmt.Println(predictedLabel[i])

        for j := 0; j < self.out; j++{

            if (predictedLabel[i][j] == 1 && actualLabel[i][j] == 1){
                accuracyCount += 1
                fmt.Println("Predicted")
            }
        }
    }
    self.AccuracyCount  = accuracyCount
    self.PredictedLabel = predictedLabel
}