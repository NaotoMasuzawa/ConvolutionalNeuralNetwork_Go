package ConvolutionPoolingLayer

import(

    "fmt"
    "math"
    "../utils"
)

type ConvolutionalPoolingLayer struct{

    imageSize [] int
    channel int
    nKernels int
    kernelSize []int
    poolSize []int
    convolvedSize []int
    pooledSize []int
    miniBatchSize int
    activationName string
    
    W [][][][] float64
    Bias [] float64
    
    Input [][][][]float64
    Convolved [][][][]float64
    Activated [][][][]float64
    Pooled [][][][]float64
    DMaxPool [][][][]float64
    DConvolve [][][][]float64
}

func Construct(self *ConvolutionalPoolingLayer, 
               imageSize      []int, 
               channel          int, 
               nKernels         int, 
               kernelSize     []int, 
               poolSize       []int, 
               convolvedSize  []int, 
               pooledSize     []int,
               miniBatchSize  int,
               activationName string){

    self.imageSize = imageSize
    self.channel = channel
    self.nKernels = nKernels
    self.kernelSize = kernelSize
    self.poolSize = poolSize
    self.convolvedSize = convolvedSize
    self.pooledSize = pooledSize
    self.miniBatchSize = miniBatchSize
    self.activationName = activationName
    
    var(
        in int = channel * kernelSize[0] * kernelSize[1]
        out int = nKernels * kernelSize[0] * kernelSize[1] / (poolSize[0] * poolSize[1])
        randBoundary float64 = math.Sqrt(6.0 / float64(in + out))
    )
        
    self.W = make([][][][]float64, nKernels)
    for i := 0; i < nKernels; i++{
        self.W[i] = make([][][]float64, channel)

        for j := 0; j < channel; j++{
            self.W[i][j] = make([][]float64, kernelSize[0])

            for k := 0; k < kernelSize[0]; k++{
                self.W[i][j][k] = make([]float64, kernelSize[1])
            }
        }
    }

    for i := 0; i < nKernels; i++{
        for j := 0; j < channel; j++{
            for k := 0; k < kernelSize[0]; k++{
                for l := 0; l < kernelSize[1]; l++{
                    self.W[i][j][k][l] = utils.Uniform(-randBoundary, randBoundary)
                }
            }
        }
    }

    self.Bias = make([] float64,nKernels)
}

func Confirm(self *ConvolutionalPoolingLayer){

    fmt.Println("ImageSize")
    fmt.Println(self.imageSize)
    fmt.Println("Channel")
    fmt.Println(self.channel)
    fmt.Println("NumberKernel")
    fmt.Println(self.nKernels)
    fmt.Println("KernelSize")
    fmt.Println(self.kernelSize)
    fmt.Println("PoolSize")
    fmt.Println(self.poolSize)
    fmt.Println("ConvolvedSize")
    fmt.Println(self.convolvedSize)
    fmt.Println("PooledSize")
    fmt.Println(self.pooledSize)
    fmt.Println("miniBatchSize")
    fmt.Println(self.miniBatchSize)
    fmt.Println("ActivationName")
    fmt.Println(self.activationName)
}

func Forward(self *ConvolutionalPoolingLayer, Input [][][][]float64, TrainOrTest string){

    if TrainOrTest == "Train"{
        self.Input = Input
    }
    Convolve(self, Input)
    MaxPool(self, Input)
}

func Backward(self *ConvolutionalPoolingLayer, prevDelta [][][][]float64, learningRate float64){

    DMaxPool(self, prevDelta, learningRate)
    DConvolve(self, learningRate)
}

func Convolve(self *ConvolutionalPoolingLayer, Input [][][][]float64){

    var convolved = make([][][][]float64, self.miniBatchSize)
    var activated = make([][][][]float64, self.miniBatchSize)
    for i := 0; i < self.miniBatchSize; i++{
        convolved[i] = make([][][]float64, self.nKernels)
        activated[i] = make([][][]float64, self.nKernels)

        for j := 0; j < self.nKernels; j++{
            convolved[i][j] = make([][]float64, self.convolvedSize[0])
            activated[i][j] = make([][]float64, self.convolvedSize[0])

            for k := 0; k < self.convolvedSize[0]; k++{
                convolved[i][j][k] = make([]float64, self.convolvedSize[1])
                activated[i][j][k] = make([]float64, self.convolvedSize[1])
            }
        }
    }

    for i := 0; i < self.miniBatchSize; i++{
        for j:= 0; j < self.nKernels; j++{
            for k := 0; k < self.convolvedSize[0]; k++{
                for l:= 0; l < self.convolvedSize[1]; l++{
                    
                    for m:= 0; m < self.channel; m++{
                        for n:= 0; n < self.kernelSize[0]; n++{
                            for o:= 0; o < self.kernelSize[1]; o++{
                                convolved[i][j][k][l] += self.W[j][m][n][o] * Input[i][m][k + n][l + o] + self.Bias[j]
                            }
                        }
                    }
                    activated[i][j][k][l] = utils.Activation(self.activationName, convolved[i][j][k][l])
                }
            }
        }
    }
    self.Convolved = convolved
    self.Activated = activated
}

func MaxPool(self *ConvolutionalPoolingLayer, Input [][][][]float64){

    var pooled = make([][][][]float64, self.miniBatchSize)
    for i := 0; i < self.miniBatchSize; i++{
        pooled[i] = make([][][]float64, self.nKernels)

        for j := 0; j < self.nKernels; j++{
            pooled[i][j] = make([][]float64, self.pooledSize[0])

            for k := 0; k < self.pooledSize[0]; k++{
                pooled[i][j][k] = make([]float64, self.pooledSize[1])
            }
        }
    }

    for i := 0; i < self.miniBatchSize; i++{
        for j := 0; j < self.nKernels; j++{
            for k := 0; k < self.pooledSize[0]; k++{
                for l := 0; l < self.pooledSize[1]; l++{
                    max := 0.0

                    for m:= 0; m < self.poolSize[0]; m++{
                        for n := 0; n < self.poolSize[1]; n++{

                            if m == 0 && n == 0{
                                max = self.Activated[i][j][k * self.poolSize[0]][l * self.poolSize[1]]
                                continue
                            }
                            if max < self.Activated[i][j][k * self.poolSize[0] + m][l * self.poolSize[1] + n]{
                                max = self.Activated[i][j][k * self.poolSize[0] + m][l * self.poolSize[1] + n]
                            }
                        }
                    }
                    pooled[i][j][k][l] = max
                }
            }
        }
    }
    self.Pooled = pooled
}

func DMaxPool(self *ConvolutionalPoolingLayer, prevDelta [][][][]float64, learningRate float64){

    var dMaxPool = make([][][][]float64, self.miniBatchSize)
    for i:= 0; i < self.miniBatchSize; i++{
        dMaxPool[i] = make([][][]float64, self.nKernels)

        for j := 0; j < self.nKernels; j++{
            dMaxPool[i][j] = make([][]float64, self.convolvedSize[0])

            for k := 0; k < self.convolvedSize[0]; k++{
                dMaxPool[i][j][k] = make([]float64, self.convolvedSize[1])
            }
        }
    }
    
    for i := 0; i < self.miniBatchSize; i++{
        for j := 0; j < self.nKernels; j++{
            for k := 0; k < self.pooledSize[0]; k++{
                for l := 0; l < self.pooledSize[1]; l++{
                    for m:= 0; m < self.poolSize[0]; m++{
                        for n := 0; n < self.poolSize[1]; n++{

                            delta := 0.0

                            if self.Pooled[i][j][k][l] == self.Activated[i][j][k * self.poolSize[0] + m][l * self.poolSize[1] + n]{
                                delta = prevDelta[i][j][k][l]
                            }
                            dMaxPool[i][j][k * self.poolSize[0] + m][l * self.poolSize[1] + n] = delta
                        }
                    }
                }
            }
        }
    }
    self.DMaxPool = dMaxPool
}

func DConvolve(self *ConvolutionalPoolingLayer, learningRate float64){

    var gradW = make([][][][]float64, self.nKernels)
    for i := 0; i < self.nKernels; i++{
        gradW[i] = make([][][]float64, self.channel)

        for j := 0; j < self.channel; j++{
            gradW[i][j] = make([][]float64, self.kernelSize[0])

            for k := 0; k < self.kernelSize[0]; k++{
                gradW[i][j][k] = make([]float64, self.kernelSize[1])
            }
        }
    }
    
    var gradBias = make([]float64, self.nKernels)

    var dConvolve = make([][][][]float64, self.miniBatchSize)
    for i := 0; i < self.miniBatchSize; i++{
        dConvolve[i] = make([][][]float64, self.channel)

        for j := 0; j < self.channel; j++{
            dConvolve[i][j] = make([][]float64, self.imageSize[0])

            for k := 0; k < self.imageSize[0]; k++{
                dConvolve[i][j][k] = make([]float64, self.imageSize[1])
            }
        }
    }

    for i := 0; i < self.miniBatchSize; i++{
        for j := 0; j < self.nKernels; j++{
            for k := 0; k < self.convolvedSize[0]; k++{
                for l := 0; l < self.convolvedSize[1]; l++{

                    d :=  self.DMaxPool[i][j][k][l] * utils.Dactivation(self.activationName, self.Convolved[i][j][k][l])
                    gradBias[j] += d

                    for m := 0; m < self.channel; m++{
                        for n := 0; n < self.kernelSize[0]; n++{
                            for o := 0; o < self.kernelSize[1]; o++{

                                gradW[j][m][n][o] += d * self.Input[i][m][k + n][l + o]
                            }
                        }
                    }
                }
            }
        }
    }

    for i := 0; i < self.nKernels; i++{
        self.Bias[i] -= learningRate * gradBias[i] / float64(self.miniBatchSize)

        for j := 0; j < self.channel; j++{
            for k := 0; k < self.kernelSize[0]; k++{
                for l := 0; l < self.kernelSize[1]; l++{

                    self.W[i][j][k][l] -= learningRate * gradW[i][j][k][l] / float64(self.miniBatchSize)
                }
            }
        }
    }

    var delta float64
    for i := 0; i < self.miniBatchSize; i++{
        for j := 0; j < self.channel; j++{
            for k := 0; k < self.imageSize[0]; k++{
                for l := 0; l < self.imageSize[1]; l++{

                    for m := 0; m < self.nKernels; m++{
                        for n := 0; n < self.kernelSize[0]; n++{
                            for o := 0; o < self.kernelSize[1]; o++{
                                
                                if (k - (self.kernelSize[0] - 1) - n < 0) || (l - (self.kernelSize[1] - 1) - o < 0){
                                    delta = 0.0
                                }else{
                                    delta = self.DMaxPool[i][m][k - (self.kernelSize[0] - 1) - n][l - (self.kernelSize[1] - 1) - o] *
                                            utils.Dactivation(self.activationName, self.Convolved[i][m][k - (self.kernelSize[0] - 1) - n][l - (self.kernelSize[1] - 1) - o]) *
                                            self.W[m][j][n][o]
                                }
                                dConvolve[i][j][k][l] += delta
                            }
                        }
                    }
                }
            }
        }
    }
    self.DConvolve = dConvolve
}