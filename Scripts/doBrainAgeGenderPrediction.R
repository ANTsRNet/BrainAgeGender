library( ANTsR )
library( ANTsRNet )
library( keras )

args <- commandArgs( trailingOnly = TRUE )

if( length( args ) != 2 )
  {
  helpMessage <- paste0( "Usage:  Rscript doBrainAgeGenderPrediction.R",
    " inputGrayMatterProbabilityImage reorientationTemplate\n" )
  stop( helpMessage )
  } else {
  inputFileName <- args[1]
  reorientTemplateFileName <- args[2]
  }

# baseDir <- "/Users/ntustison/Pkg/ANTsRNetApps/BrainAgeGender/"
# inputFileName <- paste0( baseDir, "Data/Example/exampleGrayMatter.nii.gz" )
# reorientTemplateFileName <- paste0( baseDir, "Data/Template/S_template3_resampled2.nii.gz" )

regressors <- c( "Age", "Gender/Femaleness" )
numberOfClassificationLabels <- length( regressors )

imageMods <- c( "GrayMatterProbability" )
channelSize <- length( imageMods )

cat( "Reading reorientation template", reorientTemplateFileName )
startTime <- Sys.time()
reorientTemplate <- antsImageRead( reorientTemplateFileName )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

resampledImageSize <- dim( reorientTemplate )

resnetModel <- createResNetModel3D(
  inputImageSize = c( resampledImageSize, channelSize ),
  numberOfClassificationLabels = numberOfClassificationLabels,
  mode = "regression" )

cat( "Loading weights file" )
startTime <- Sys.time()
weightsFileName <- "resnetModelWeights.h5"  # getPretrainedNetwork( "brainAgeGender" )
load_model_weights_hdf5( resnetModel, filepath = weightsFileName )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

resnetModel %>% compile( loss = 'mse',
  optimizer = optimizer_adam( lr = 0.0001 ),
  metrics = list( "mean_absolute_error" ) )

# Process input

startTimeTotal <- Sys.time()

cat( "Reading ", inputFileName )
startTime <- Sys.time()
image <- antsImageRead( inputFileName, dimension = 3 )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

cat( "Normalizing to template (developer note:  probably should switch to a quick rigid/affine registration" )
startTime <- Sys.time()
centerOfMassTemplate <- getCenterOfMass( reorientTemplate )
centerOfMassImage <- getCenterOfMass( image )
xfrm <- createAntsrTransform( type = "Euler3DTransform",
  center = centerOfMassTemplate,
  translation = centerOfMassImage - centerOfMassTemplate )
warpedImage <- applyAntsrTransformToImage( xfrm, image, reorientTemplate )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

batchX <- array( data = as.array( warpedImage ),
  dim = c( 1, resampledImageSize, channelSize ) )
batchX <- ( batchX - mean( batchX ) ) / sd( batchX )

cat( "Prediction and decoding" )
startTime <- Sys.time()
predictedData <- resnetModel %>% predict( batchX, verbose = 0 )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( " (elapsed time:", elapsedTime, "seconds)\n" )

cat( "\n\n********************************\n" )
cat( "Predicted age: ", predictedData[, 1], "years\n" )
cat( "Predicted gender: ", predictedData[, 2], " ---> [0, 1] = [male, female]\n" )
cat( "********************************\n\n" )

endTimeTotal <- Sys.time()
elapsedTimeTotal <- endTimeTotal - startTimeTotal
cat( "\nTotal elapsed time:", elapsedTimeTotal, "seconds\n\n" )
