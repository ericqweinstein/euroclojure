(ns ^{:doc    "Convolutional neural net for image recognition."
      :author "Eric Weinstein <eric.q.weinstein@gmail.com>"}
euroclojure.convnet
  (:import (org.deeplearning4j.nn.conf NeuralNetConfiguration$Builder)
           (org.deeplearning4j.nn.multilayer MultiLayerNetwork)
           (org.deeplearning4j.nn.api OptimizationAlgorithm)
           (org.deeplearning4j.nn.conf Updater GradientNormalization LearningRatePolicy)
           (org.deeplearning4j.nn.conf.layers ConvolutionLayer$Builder
                                              SubsamplingLayer$Builder
                                              DenseLayer$Builder
                                              OutputLayer$Builder LocalResponseNormalization LocalResponseNormalization$Builder)
           (org.deeplearning4j.optimize.listeners ScoreIterationListener)
           (org.deeplearning4j.nn.weights WeightInit)
           (org.deeplearning4j.datasets.datavec RecordReaderDataSetIterator)
           (org.deeplearning4j.datasets.iterator MultipleEpochsIterator)

           (org.nd4j.linalg.lossfunctions LossFunctions$LossFunction)
           (org.nd4j.linalg.dataset.api.preprocessor ImagePreProcessingScaler)

           (org.datavec.api.io.labels ParentPathLabelGenerator)
           (org.datavec.image.loader NativeImageLoader)
           (org.datavec.api.split FileSplit)
           (org.datavec.api.io.filters BalancedPathFilter)
           (org.datavec.image.recordreader ImageRecordReader)

           (java.io File)
           (java.util Random)
           (org.deeplearning4j.nn.conf.distribution GaussianDistribution NormalDistribution)))

;; Effectively constants for our neural network setup.
(def height 100)
(def width 100)
(def channels 3)
(def seed (long 42))
(def rng (Random. seed))
(def num-examples 80)
(def num-labels 4)
(def batch-size 20)
(def split-train-test 0.8)
(def epochs 50)
(def num-cores 2)

(defn conv-init
  "Creates an initial layer."
  [kernel stride pad out]
  (-> (ConvolutionLayer$Builder. kernel stride pad)
      (.name "cnn1")
      (.nIn channels)
      (.nOut out)
      (.biasInit 0)
      (.build)))

(def output-layer
  (-> (OutputLayer$Builder. (LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD))
      (.name "output")
      (.nOut 4)
      (.activation "softmax")
      (.build)))

(defn conv-3x3
  "Creates a 3x3 convolutional layer."
  [name out bias]
  (-> (ConvolutionLayer$Builder.
        (int-array [3 3])
        (int-array [1 1])
        (int-array [3 3]))
      (.name name)
      (.nOut out)
      (.biasInit bias)
      (.build)))

(defn lrn
  "Creates a new normalization layer."
  [name]
  (-> (LocalResponseNormalization$Builder.)
      (.name name)
      (.build)))

(defn conv-5x5
  "Creates a 5x5 convolutional layer."
  [name out stride pad bias]
  (-> (ConvolutionLayer$Builder. (int-array [5 5]) stride pad)
      (.name name)
      (.nOut out)
      (.biasInit bias)
      (.build)))

(defn max-pool
  "Creates a max pooling layer."
  [name kernel]
  (-> (SubsamplingLayer$Builder. kernel (int-array [2 2]))
      (.name name)
      (.build)))

(defn fully-connected
  "Creates a dense fully-connected layer."
  [out]
  (-> (DenseLayer$Builder.)
      (.nOut out)
      (.build)))

(def nn-conf
  (-> (NeuralNetConfiguration$Builder.)
      (.seed seed)
      (.iterations 1)
      (.regularization false)
      (.l2 0.005)
      (.activation "relu")
      (.learningRate 0.0001)
      (.weightInit (WeightInit/XAVIER))
      (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
      (.updater Updater/RMSPROP)
      (.momentum 0.9)
      (.list)
      (.layer 0 (conv-init (int-array [5 5]) (int-array [1 1]) (int-array [0 0]) 50))
      (.layer 1 (max-pool "maxpool1" (int-array [2 2])))
      (.layer 2 (conv-5x5 "cnn2" 100 (int-array [5 5]) (int-array [1 1]) 0))
      (.layer 3 (max-pool "maxpool2" (int-array [2 2])))
      (.layer 4 (fully-connected 500))
      (.layer 5 output-layer)
      (.backprop true)
      (.pretrain false)
      (.cnnInputSize 100 100 channels)
      (.build)))

(comment
  (defn alternate-fully-connected
  "Creates a fully-connected layer for the more complex NN conf."
  [name out]
  (-> (DenseLayer$Builder.)
      (.name name)
      (.nOut out)
      (.biasInit 1)
      (.dropOut 0.5)
      (.dist (GaussianDistribution. 0 0.005))
      (.build)))

  (def alternate-nn-conf
    (-> (NeuralNetConfiguration$Builder.)
        (.seed seed)
        (.weightInit WeightInit/DISTRIBUTION)
        (.dist (NormalDistribution. 0.0 0.01))
        (.activation "relu")
        (.updater Updater/NESTEROVS)
        (.iterations 1)
        (.gradientNormalization GradientNormalization/RenormalizeL2PerLayer)
        (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
        (.learningRate 0.01)
        (.biasLearningRate 0.0001)
        (.learningRateDecayPolicy LearningRatePolicy/Step)
        (.lrPolicyDecayRate 0.1)
        (.lrPolicySteps 100000)
        (.regularization true)
        (.l2 0.0005)
        (.momentum 0.9)
        (.miniBatch false)
        (.list)
        (.layer 0 (conv-init (int-array [11 11]) (int-array [4 4]) (int-array [3 3]) 96))
        (.layer 1 (lrn "lrn1"))
        (.layer 2 (max-pool "maxpool1" (int-array [3 3])))
        (.layer 3 (conv-5x5 "cnn2" 256 (int-array [1 1]) (int-array [2 2]) 1))
        (.layer 4 (lrn "lrn2"))
        (.layer 5 (max-pool "maxpool2" (int-array [3 3])))
        (.layer 6 (conv-3x3 "cnn3" 384 0))
        (.layer 7 (conv-3x3 "cnn4" 384 1))
        (.layer 8 (conv-3x3 "cnn5" 256 1))
        (.layer 9 (max-pool "maxpool3" (int-array [3 3])))
        (.layer 10 (alternate-fully-connected "ffn1" 4096))
        (.layer 11 (alternate-fully-connected "ffn2" 4096))
        (.layer 12 output-layer)
        (.backprop true)
        (.pretrain false)
        (.cnnInputSize height width channels)
        (.build)))
)

(def label-maker
  (ParentPathLabelGenerator.))

(def main-path
  (File. (System/getProperty "user.dir" "resources/animals/")))

(def file-split
  (FileSplit. main-path NativeImageLoader/ALLOWED_FORMATS rng))

(def path-filter
  (BalancedPathFilter. rng label-maker num-examples num-labels batch-size))

(def splits
  (.sample file-split path-filter
           (double-array
             (* num-examples (+ 1 split-train-test))
             (* num-examples (- 1 split-train-test)))))

(def train-data (first splits))

(def test-data (last splits))

(def scaler (ImagePreProcessingScaler. 0 1))

(defn build-network
  "Builds our neural network."
  []
  (println "Building network...")
  (MultiLayerNetwork. nn-conf)) ; other-nn

(defn make-record-reader
  "Makes an ImageRecordReader and initializes it."
  [data]
  (let [record-reader (ImageRecordReader. height width channels label-maker)]
    (.initialize record-reader data)
    record-reader))

(defn train
  "Trains the deep neural network."
  []
  (let [network       (build-network)
        record-reader (make-record-reader train-data)
        data-iter     (RecordReaderDataSetIterator. record-reader batch-size 1 num-labels)]

    (println "Initializing network...")
    (-> network
        (.init))
        ;; TODO: Fix NPE that occurs when setting listeners. (EQW 28 Nov 2016)
        ;; (.setListeners (ScoreIterationListener. 1)))

    (println "Training model...")
    (.fit scaler data-iter)
    (.setPreProcessor data-iter scaler)
    (.fit network (MultipleEpochsIterator. epochs data-iter num-cores))

    network))

(defn evaluate
  "Evaluates the effectiveness of our model."
  [network]
  (let [record-reader (make-record-reader test-data)
        data-iter     (RecordReaderDataSetIterator. record-reader batch-size 1 num-labels)]
    (println "Evaluating model...")
    (.fit scaler data-iter)
    (.setPreProcessor data-iter scaler)
    (println (-> network
                 (.evaluate data-iter)
                 (.stats true)))))

(defn run
  "Runs our neural network."
  []
  (evaluate (train)))
