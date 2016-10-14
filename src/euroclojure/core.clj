(ns ^{:doc    "Machine learning code for EuroClojure 2016."
      :author "Eric Weinstein <eric.q.weinstein@gmail.com>"}
  euroclojure.core
  (:require [euroclojure.flambo :as flambo]
            [euroclojure.convnet :as convnet]))

(defn -main
  "The main entrypoint for our code."
  [& _]
  (println "*******************************************")
  (println "Decision tree accuracy: " (- 1.0 (float flambo/test-err)))
  (println "*******************************************")
  (convnet/run))