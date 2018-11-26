# import the necessary packages
from myPackage.ir.bagofvisualwords import BagOfVisualWords
from myPackage.indexer.bovwindexer import BOVWIndexer
import pickle
import h5py

dbPath = "database/features.hdf5" # database path
codebookPath = "database/vocab.p"
bovw_db = "database/bovw.hdf5" # bovw database path
maxBufferSize = 500

# load the codebook vocabulary and initialize the bag-of-visual-words transformer
vocab = pickle.load(open(codebookPath, "rb"))
bovw = BagOfVisualWords(vocab)

# open the features database and initialize the bag-of-visual-words indexer
featuresDB = h5py.File(dbPath, mode="r")
bi = BOVWIndexer(bovw.codebook.shape[0], bovw_db,
	estNumImages=featuresDB["image_ids"].shape[0],
	maxBufferSize=maxBufferSize)

# loop over the image IDs and index
for (i, (imageID, offset)) in enumerate(zip(featuresDB["image_ids"], featuresDB["index"])):
	# extract the feature vectors for the current image using the starting and
	# ending offsets (while ignoring the keypoints) and then quantize the
	# features to construct the bag-of-visual-words histogram
	features = featuresDB["features"][offset[0]:offset[1]][:, 2:]
	hist = bovw.describe(features)

	# normalize the histogram such that it sums to one then add the
	# bag-of-visual-words to the index
	hist /= hist.sum()
	bi.add(hist)

# close the features database and finish the indexing process
featuresDB.close()
bi.finish()