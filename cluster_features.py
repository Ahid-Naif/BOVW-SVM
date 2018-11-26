# import the necessary packages
from myPackage.ir.vocabulary import Vocabulary
import pickle

dbPath = "database/features.hdf5" # database path
numClusters = 512
samplePercenrage = 0.25
codebookPath = "database/vocab.p"

# create the visual words vocabulary
voc = Vocabulary(dbPath)
vocab = voc.fit(numClusters, samplePercenrage)

# dump the clusters to file
print("[INFO] storing cluster centers...")
pickle.dump(vocab, open(codebookPath, "wb"))