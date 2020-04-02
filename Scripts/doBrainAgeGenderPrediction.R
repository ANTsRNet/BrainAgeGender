library( ANTsR )
library( ANTsRNet )
library( keras )

args <- commandArgs( trailingOnly = TRUE )

verbose <- TRUE

if( length( args ) == 1 )
  {
  helpMessage <- paste0( "Usage:  Rscript doBrainAgePrediction.R outputCsvFile inputT1_1 inputT1_2 inputT1_3 ...\n" )
  stop( helpMessage )
  } else {
  outputCsvFile <- args[1]
  inputFileName <- args[2:length( args )]
  }

#################
#
#  Brain extraction function adapted from
#      https://github.com/ANTsXNet/BrainExtraction
#

brainExtraction <- function( image, verbose = TRUE )
  {
  classes <- c( "background", "brain" )
  numberOfClassificationLabels <- length( classes )
  imageMods <- c( "T1" )
  channelSize <- length( imageMods )

  reorientTemplateFileName <- paste0( getwd(), "/S_template3_resampled.nii.gz" )
  if( ! file.exists( reorientTemplateFileName ) )
    {
    if( verbose == TRUE )
      {
      cat( "Brain extraction:  downloading template.\n" )
      }
    reorientTemplateUrl <- "https://github.com/ANTsXNet/BrainAgeGender/blob/master/Data/Template/S_template3_resampled.nii.gz?raw=true"
    download.file( reorientTemplateUrl, reorientTemplateFileName, quiet = !verbose )
    }
  reorientTemplate <- antsImageRead( reorientTemplateFileName )
  resampledImageSize <- dim( reorientTemplate )

  unetModel <- createUnetModel3D( c( resampledImageSize, channelSize ),
    numberOfOutputs = numberOfClassificationLabels,
    numberOfLayers = 4, numberOfFiltersAtBaseLayer = 8, dropoutRate = 0.0,
    convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
    weightDecay = 1e-5 )

  weightsFileName <- paste0( getwd(), "/brainExtractionWeights.h5" )
  if( ! file.exists( weightsFileName ) )
    {
    if( verbose == TRUE )
      {
      cat( "Brain extraction:  downloading model weights.\n" )
      }
    weightsFileName <- getPretrainedNetwork( "brainExtraction", weightsFileName )
    }
  unetModel$load_weights( weightsFileName )

  if( verbose == TRUE )
    {
    cat( "Brain extraction:  normalizing image to the template.\n" )
    }
  centerOfMassTemplate <- getCenterOfMass( reorientTemplate )
  centerOfMassImage <- getCenterOfMass( image )
  xfrm <- createAntsrTransform( type = "Euler3DTransform",
    center = centerOfMassTemplate,
    translation = centerOfMassImage - centerOfMassTemplate )
  warpedImage <- applyAntsrTransformToImage( xfrm, image, reorientTemplate )

  batchX <- array( data = as.array( warpedImage ),
    dim = c( 1, resampledImageSize, channelSize ) )
  batchX <- ( batchX - mean( batchX ) ) / sd( batchX )

  if( verbose == TRUE )
    {
    cat( "Brain extraction:  prediction and decoding.\n" )
    }
  predictedData <- unetModel %>% predict( batchX, verbose = 0 )
  probabilityImagesArray <- decodeUnet( predictedData, reorientTemplate )

  if( verbose == TRUE )
    {
    cat( "Brain extraction:  renormalize probability mask to native space.\n" )
    }
  probabilityImage <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
    probabilityImagesArray[[1]][[2]], image )

  return( probabilityImage )
  }

#################
#
#  Preprocessing function
#      * Denoising
#      * N4 bias correction
#      * histogram or regression intensity matching
#

antsPreprocessImage <- function( image, mask = NULL, doBiasCorrection = TRUE,
  doDenoising = TRUE, referenceImage = NULL, matchingType = c( "regression", "histogram" ),
  verbose = TRUE )
  {
  preprocessedImage <- image

  # Do bias correction
  if( doBiasCorrection == TRUE )
    {
    if( verbose == TRUE )
      {
      cat( "Preprocessing:  bias correction.\n" )
      }
    preprocessedImage <- n4BiasFieldCorrection( image, mask, shrinkFactor = 4, verbose = verbose )
    }

  # Do denoising
  if( doDenoising == TRUE )
    {
    if( verbose == TRUE )
      {
      cat( "Preprocessing:  denoising.\n" )
      }
    preprocessedImage <- denoiseImage( preprocessedImage, mask, shrinkFactor = 1, verbose = verbose )
    }

  # Do image matching
  if( ! is.null( referenceImage ) )
    {
    if( verbose == TRUE )
      {
      cat( "Preprocessing:  intensity matching.\n" )
      }
    if( matchingType == "regression" )
      {
      preprocessedImage <- regressionMatchImage( preprocessedImage, referenceImage )
      } else if( matchingType == "histogram" ) {
      preprocessedImage <- histogramMatchImage( preprocessedImage, referenceImage )
      } else {
      stop( paste0( "Error:  unrecognized match type = ", matchingType, "\n" )
      }
    }
  return( preprocessedImage )
  }

#################
#
#  Data augmentation
#

brainAgeDataAugmentation <- function( image, imageSubsampled, patchSize = 96,
  batchSize = 1, affineStd = 0.01, verbose = TRUE )
  {
  # Channel 1: original image/patch
  # Channel 2: difference image/patch with MNI average
  numberOfChannels <- 2

  imageOffset <- 10
  imageDimensions <- dim( image )
  imageSubsampledDimensions <- dim( imageSubsampled )

  mniImageFileName <- paste0( getwd() "/mniAverage.nii.gz" )
  if( ! file.exists( mniImageFileName ) )
    {
    if( verbose == TRUE )
      {
      cat( "Data augmentation:  downloading MNI average image.\n" )
      }
    mniUrl <- "https://github.com/ANTsXNet/BrainAgeGender/blob/master/Data/Template/mniAverage.nii.gz?raw=true"
    download.file( mniUrl, mniImageFileName, quiet = !verbose )
    }
  mniAverage <- antsImageRead( mniImageFileName )

  mniImageSubsampledFileName <- paste0( getwd() "/mniAverageSubsampled.nii.gz" )
  if( ! file.exists( mniImageSubsampledFileName ) )
    {
    if( verbose == TRUE )
      {
      cat( "Data augmentation:  downloading MNI average image.\n" )
      }
    mniUrl <- "https://github.com/ANTsXNet/BrainAgeGender/blob/master/Data/Template/mniAverageSubsampled.nii.gz?raw=true"
    download.file( mniUrl, mniImageSubsampledFileName, quiet = !verbose )
    }
  mniAverageSubsampled <- antsImageRead( mniImageFileName )

  imageDifference <- image - mniAverage
  imageSubsampledDifference <- imageSubsampled - mniAverageSubsampled

  imageArray <- array( data = NA, dim = c( batchSize, imageSubsampledDimensions, numberOfChannels ) )
  patchArray <- array( data = NA, dim = c( batchSize, rep( patchSize, 3 ), numberOfChannels ) )

  randomImages <- randomImageTransformAugementation( imageSubsampled,
    interpolator = c( "linear","linear" ), list( list( image, imageDifference ) ),
    list( imageDifference ), sdAffine = affineStd, n = batchSize, normalization = "01" )

  for( i in seq_len( batchSize ) )
    {
    lowerIndices <- rep( NA, 3 )
    for( d in seq_len( 3 ) )
      {
      lowerIndices[d] <- sample.int( imageOffset:( imageDimensions[d] - patchSize - imageOffset ), 1 )
      }
    upperIndices <- lowerIndices + rep( patchSize, 3 ) - 1
    patch <- cropIndices( image, lowerIndices, upperIndices )
    patchDifference <- cropIndices( imageDifference, lowerIndices)

    imageArray[i,,,,1] <- as.array( imageSubsampled )
    imageArray[i,,,,2] <- randomImages$outputPredictorList[[i]][[2]]
    patchArray[i,,,,1] <- patch
    patchArray[i,,,,2] <- patchDifference
    }
  return( list( imageArray, patchArray ) )
  }

#################
#
#  Main routine
#

verbose <- TRUE

targetTemplateDimension <- c( 192L, 224L, 192L )

channelSize <- 2L
patchSize <- c( rep( 96L, 3L ), channelSize )

# Prepare the template

templateFileName <- paste0( getwd(), "/template_brainAge.nii.gz" )
if( ! file.exists( templateFileName ) )
  {
  if( verbose == TRUE )
    {
    cat( "Brain age:  downloading template.\n" )
    }
  templateUrl <- "https://github.com/ANTsXNet/BrainExtraction/blob/master/Data/Template/S_template3_resampled.nii.gz?raw=true"
  download.file( templateUrl, templateFileName, quiet = !verbose )
  }
originalTemplate <- antsImageRead( templateFileName )
template <- resampleImage( originalTemplate, targetTemplateDimension,
  useVoxels = TRUE, interpType = "linear" )
templateNormalized <- template %>% iMath( "Normalize" )
templateProbabilityMask <- brainExtraction( template, verbose = verbose )
templateSubsampled <- resampleImage( template,
  as.integer( floor( targetTemplateDimension / 2 ) ), useVoxels = TRUE,
  interpType = "linear" )

# Prepare the model and load the weights

classes <- c( "Site", "Age", "Gender" )
numberOfClasses <- as.integer( channelSize * length( classes ) )
siteNames <- c( "DLBS", "HCP", "IXI", "NKIRockland", "OAS1_", "SALD" )

inputImageSize = c( dim( templateSubsampled ), channelSize )
resnetModel <- createResNetModel3D( inputImageSize,
  numberOfClassificationLabels = 1000, layers = 1:4,
  residualBlockSchedule = c(3, 4, 6, 3),
  lowestResolution = 64, cardinality = 64,
  mode = "classification")
penultimateLayerName <- as.character(
  resnetModel$layers[[length( resnetModel$layers ) - 1]]$name )
siteLayer <- layer_dense( get_layer( resnetModel, penultimateLayerName )$output,
  units = numberOfClasses, activation = "sigmoid" )
ageLayer <- layer_dense( get_layer( resnetModel, penultimateLayerName )$output,
  units = 1L, activation = "linear" )
genderLayer <- layer_dense( get_layer( resnetModel, penultimateLayerName )$output,
  units = 1L, activation = "sigmoid" )

inputPatch <- layer_input( patchShape )
model <- keras_model( inputs = list( resnetModel$input, inputPatch ) ),
  outputs = list( siteLayer, ageLayer, genderLayer )

weightsFileName <- paste0( getwd(), "/resNet4LayerLR64Card64b.h5" )
if( ! file.exists( weightsFileName ) )
  {
  if( verbose == TRUE )
    {
    cat( "Brain age:  downloading model weights file.\n" )
    }
  weightsFileName <- getPretrainedNetwork( "brainAgeGender", weightsFileName )
  }
load_model_weights_hdf5( model, weightsFileName )

brainAges <- rep( NA, length( inputFileNames ) )
brainGenders <- rep( NA, length( inputFileNames ) )
for( i in seq_len( length( inputFileNames ) ) )
  {
  inputImage <- antsImageRead( inputFileNames[i] )
  if( verbose )
    {
    cat( "Preprocessing input image ", inputFileNames[i], ".\n" )
    }
  inputImage <- antsPreprocessImage( inputImage )
  if( verbose )
    {
    cat( "Brain extraction.\n" )
    }
  inputProbabilityBrainMask <- brainExtraction( inputImage, verbose = TRUE )
  inputBrainMask <- thresholdImage( inputProbabilityBrainMask, 0.5, Inf )
  inputBrain <- inputBrainMask * inputImage
  inputBrainNormalized <- inputBrain %>% iMath( "Normalize" )

  if( verbose )
    {
    cat( "Registration to template.\n" )
    }
  templatexInputRegistration <- antsRegistration( fixed = templateNormalized,
    moving = inputBrainNormalized, typeofTransform = "Affine", verbose = verbose )

  inputImageWarped <- antsApplyTransforms( template, inputImage,
    templatexInputRegistration$fwdtransforms, interpolator = "linear" )
  inputImageWarped <- inputImageWarped %>% iMath( "Normalize" )
  inputImageWarpedSubsampled <- antsApplyTransforms( templateSubsampled, inputImage,
    templatexInputRegistration$fwdtransforms, interpolator = "linear"  )
  inputImageWarpedSubsampled <- inputImageWarpedSubsampled %>% iMath( "Normalize" )

  augmentation <- brainAgeDataAugmentation( inputImageWarped, inputImageWarpedSubsampled,
    batchSize = batchSize, affineStd = 0.01, verbose = TRUE )
  predictions <- predict( model, augmentation )

  # siteDataFrame <- data.frame( matrix( predictions[[1]], ncol = length( siteNames ) ) )
  # colnames( siteDataFrame ) <- siteNames
  # for( k in seq_len( nrow( siteDataFrame ) ) )
  #   {
  #   siteDataFrame[k,] <- siteDataFrame[k,] / sum( siteDataFrame[k,] )
  #   }

  brainAges[i] <- as.numeric( predictions[[2]] )
  brainGenders[i] <- as.numeric( predictions[[3]] )
  }

brainAgeDataFrame <- data.frame( FileName = inputFileNames, Age = brainAges,
  Gender = brainGenders )
write.csv( brainAgeDataFrame, file = outputCsvFile, row.names = FALSE )




