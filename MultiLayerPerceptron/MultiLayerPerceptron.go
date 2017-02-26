package MultiLayerPerceptron

import(

    "fmt"
    "math/rand"
    "../utils"
)

type MultiLayerPerceptron struct{

    In int
    Out int
    dropOut bool
    dropOutPossibility float64
    miniBatchSize int
    activationName string

    W [][]float64
    Bias []float64
    
    Input [][]float64
    PreActivate [][]float64
    Activated [][]float64
    Delta [][]float64
    DropOutMask [][]int
}

func Construct(self *MultiLayerPerceptron, 
               In                 int,
               Out                int,
               dropOut            bool,
               dropOutPossibility float64,
               miniBatchSize      int,
               activationName     string){

    self.In = In
    self.Out = Out
    self.dropOut = dropOut
    self.dropOutPossibility = dropOutPossibility
    self.miniBatchSize = miniBatchSize
    self.activationName = activationName

    var randomBoundary float64 = 1.0 / float64(In)
    self.W = make([][]float64, Out)
    for i := 0; i < Out; i++{
        self.W[i] = make([]float64, In)
    }
    for i := 0; i < Out; i++{
        for j := 0; j < In; j++{
            self.W[i][j] = utils.Uniform(-randomBoundary, randomBoundary)
        }
    }

    self.Bias = make([]float64, Out)
}

func Confirm(self *MultiLayerPerceptron){

    fmt.Println("In")
    fmt.Println(self.In)
    fmt.Println("Out")
    fmt.Println(self.Out)
    fmt.Println("Dropout")
    fmt.Println(self.dropOut)
    fmt.Println("DropOutPossibility")
    fmt.Println(self.dropOutPossibility)
    fmt.Println("miniBatchSize")
    fmt.Println(self.miniBatchSize)
    fmt.Println("ActivationName")
    fmt.Println(self.activationName)
}

func Forward(self *MultiLayerPerceptron, input [][]float64, TrainOrTest string){

    if TrainOrTest == "Train"{
        self.Input = input
    }
    
    Output(self, input, TrainOrTest)

    if self.dropOut == true{
        dropOut(self, TrainOrTest)
    }
}

func Backward(self *MultiLayerPerceptron, prevDelta [][]float64, prevW [][] float64, prevLayerOut int, learningRate float64){

    var gradW = make([][]float64, self.Out)
    for i:= 0; i < self.Out; i++{
        gradW[i] = make([]float64, self.In)
    }

    var gradBias = make([]float64, self.Out)

    var delta = make([][]float64, self.miniBatchSize)
    for i := 0; i < self.miniBatchSize; i++{
        delta[i] = make([]float64, self.Out)
    }

    for i := 0; i < self.miniBatchSize; i++{
        for j := 0; j < self.Out; j++{
            for k:= 0; k < prevLayerOut; k++{

                if self.dropOut == true{
                    delta[i][j] += float64(self.DropOutMask[i][j]) * prevW[k][j] * prevDelta[i][k]
                }else if self.dropOut == false{
                    delta[i][j] += prevW[k][j] * prevDelta[i][k]
                }
                delta[i][j] *= utils.Dactivation(self.activationName, self.PreActivate[i][j])

                gradBias[j] += delta[i][j]

                for l := 0; l < self.In; l++{
                    gradW[j][l] += delta[i][j] * self.Input[i][l]
                }
            }
        }
    }

    for i := 0; i < self.Out; i++{
        for j := 0; j < self.In; j++{
            self.W[i][j] -= learningRate * gradW[i][j] / float64(self.miniBatchSize)
        }
        self.Bias[i] -= learningRate * gradBias[i] / float64(self.miniBatchSize)
    }
    self.Delta = delta
}

func Output(self *MultiLayerPerceptron, input [][]float64, TrainOrTest string){

    var preActivate = make([][]float64, self.miniBatchSize)
    var activated = make([][]float64, self.miniBatchSize)
    for i := 0; i < self.miniBatchSize; i++{
        preActivate[i] = make([]float64, self.Out)
        activated[i] = make([]float64, self.Out)
    }

    for i := 0; i < self.miniBatchSize; i++{
        for j := 0; j < self.Out; j++{
            Out := 0.0

            for k := 0; k < self.In; k++{
                Out += self.W[j][k] * input[i][k]
            }
            preActivate[i][j] = Out + self.Bias[j]
            activated[i][j] = utils.Activation(self.activationName, preActivate[i][j])
        }
    }
    self.PreActivate = preActivate
    self.Activated = activated
}

func dropOut(self *MultiLayerPerceptron, TrainOrTest string){

    var dropOutMask = make([][]int, self.miniBatchSize)
    for i := 0; i < self.miniBatchSize; i++{
        dropOutMask[i] = make([]int, self.Out)
    }

    if TrainOrTest == "Train"{
        for i := 0; i < self.miniBatchSize; i++{
            for j := 0; j < self.Out; j++{
                random := rand.Float64()
                
                if random < self.dropOutPossibility{
                    dropOutMask[i][j] = 0
                    self.Activated[i][j] *= float64(dropOutMask[i][j])
                }else{
                    dropOutMask[i][j] = 1
                    self.Activated[i][j] *= float64(dropOutMask[i][j])
                }
            }
        }
    }else if TrainOrTest == "Test"{
        for i := 0; i < self.miniBatchSize; i++{
            for j := 0; j < self.Out; j++{
                self.Activated[i][j] *= (1 - self.dropOutPossibility)
            }
        }
    }
    self.DropOutMask = dropOutMask
}