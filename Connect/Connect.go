package Connect

type Connect struct{

    miniBatchSize int
    nKernel int
    pooledSize []int
    flattenedSize int
    hiddenLayerSize int

    Flattened [][]float64
    Unflattened [][][][]float64
}

func Construct(self *Connect,
               miniBatchSize   int,
               nKernel         int,
               pooledSize []   int,
               flattenedSize   int,
               hiddenLayerSize int){

    self.miniBatchSize   = miniBatchSize
    self.nKernel         = nKernel
    self.pooledSize      = pooledSize
    self.flattenedSize   = flattenedSize
    self.hiddenLayerSize = hiddenLayerSize
}

func Flatten(self *Connect, input [][][][]float64){

    var flattened = make([][]float64, self.miniBatchSize)
    for i := 0; i < self.miniBatchSize; i++{
        flattened[i] = make([]float64, self.flattenedSize)
    }

    for i := 0; i < self.miniBatchSize; i++{
        index := 0
        
        for j := 0; j < self.nKernel; j++{
            for k := 0; k < self.pooledSize[0]; k++{
                for l := 0; l < self.pooledSize[1]; l++{

                    flattened[i][index] = input[i][j][k][l]
                    index += 1
                }
            }
        }
    }
    self.Flattened = flattened
}

func Unflatten(self *Connect, input [][]float64, W [][]float64){

    var Delta [][]float64
    Delta = make([][]float64, self.miniBatchSize)
    for i := 0; i < self.miniBatchSize; i++{
        Delta[i] = make([]float64, self.flattenedSize)
    }

    var unflattened = make([][][][]float64, self.miniBatchSize)
    
    for i := 0; i < self.miniBatchSize; i++{
        unflattened[i] = make([][][]float64, self.nKernel)

        for j := 0; j < self.nKernel; j++{
            unflattened[i][j] = make([][]float64, self.pooledSize[0])

            for k := 0; k < self.pooledSize[0]; k++{
                unflattened[i][j][k] = make([]float64, self.pooledSize[1])
            }
        }
    }

    for i := 0; i < self.miniBatchSize; i++{
        for j := 0; j < self.flattenedSize; j++{
            for k := 0; k < self.hiddenLayerSize; k++{

                Delta[i][j] = W[k][j] * input[i][k]
            }
        }
    }

    for i := 0; i < self.miniBatchSize; i++{
        index := 0

        for j := 0; j < self.nKernel; j++{
            for k := 0; k < self.pooledSize[0]; k++{
                for l := 0; l < self.pooledSize[1]; l++{
                    unflattened[i][j][k][l] = Delta[i][index]
                    index += 1
                }
            }
        }
    }
    self.Unflattened = unflattened
}