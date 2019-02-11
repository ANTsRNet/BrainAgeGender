library( ANTsR )
library( ANTsRNet )
library( keras )
library( tensorflow )

keras::backend()$clear_session()

Sys.setenv( "CUDA_VISIBLE_DEVICES" = 3 )

imageMods <- c( "GrayMatter" )
channelSize <- length( imageMods )
batchSize <- 8L

baseDirectory <- '/home/ntustison/Data/'
scriptsDirectory <- paste0( baseDirectory, '/Scripts/BrainAge/' )
source( paste0( scriptsDirectory, 'brainAgeBatchGenerator.R' ) )

templateDirectory <- paste0( baseDirectory, 'Templates/' )
reorientTemplateDirectory <- paste0( templateDirectory, '/Kirby/SymmetricTemplate/' )
reorientTemplate <- antsImageRead( paste0( reorientTemplateDirectory, "S_template3_resampled2.nii.gz" ) )

dataDirectories <- c()

dataSets <- c( "IXI", "Kirby", "NKI", "Oasis" )

brainImageFiles <- c()
ages <- c()
gender <- c()
for( i in seq_len( length( dataSets ) ) )
  {
  demographics <- read.csv( paste0( baseDirectory, "CorticalThicknessData2014/", dataSets[i], ".csv" ) )
  dataDirectory <- paste0( baseDirectory, "CorticalThicknessData2014/", dataSets[i], "/ThicknessAnts/" )

  localAges <- demographics$Age
  if( is.null( localAges ) )
    {
    localAges <- demographics$AGE
    }
  localIds <- demographics[,1]

  genderScale <- 10

  # recast male/female to 0/genderScale
  localGender <- c()
  if( i == 1 )                # IXI
    {
    localGender <- demographics$SEX_ID
    localGender <- (localGender - 1) * genderScale
    } else if( i == 2 ) {     # Kirby
    localGender <- demographics$M.F
    localGender[which( localGender == 'M' )] <- 0
    localGender[which( localGender == 'F' )] <- 1 * genderScale
    } else if( i == 3 ) {     # NKI
    localGender <- demographics$Gender
    localGender[which( localGender == 'male' )] <- 0
    localGender[which( localGender == 'female' )] <- 1 * genderScale
    } else if( i == 4 ) {     # Oasis
    localGender <- demographics$M.F
    localGender[which( localGender == 'M' )] <- 0
    localGender[which( localGender == 'F' )] <- 1 * genderScale
    }

  for( j in seq_len( length( localIds ) ) )
    {

    if( is.numeric( localAges[j] ) && !is.na( localAges[j] ) )
      {
      imageFiles <- c()
      if( i == 1 )
        {
        ixiId <- formatC( localIds[j], width = 3, format = "d", flag = "0" )
        imageFiles <- list.files( path = dataDirectory,
          pattern = paste0( 'IXI', ixiId, "-", ".*BrainSegmentationPosteriors2.nii.gz" ),
          full.names = TRUE, recursive = TRUE )
        } else {
        imageFiles <- list.files( path = dataDirectory,
          pattern = paste0( localIds[j], ".*BrainSegmentationPosteriors2.nii.gz" ),
          full.names = TRUE, recursive = TRUE )
        }
      if( length( imageFiles ) > 0 )
        {
        ages <- append( ages, localAges[j] )
        gender <- append( gender, localGender[j] )
        brainImageFiles <- append( brainImageFiles, imageFiles[1] )
        }
      }
    }
  }
gender <- as.numeric( gender )

regressors <- cbind( ages, gender )
# regressors <- ages

trainingImageFiles <- list()
trainingMaskFiles <- list()
trainingTransforms <- list()

missingFiles <- c()

cat( "Loading data...\n" )
pb <- txtProgressBar( min = 0, max = length( brainImageFiles ), style = 3 )

count <- 1
for( i in seq_len( length( brainImageFiles ) ) )
  {
  setTxtProgressBar( pb, i )

  subjectId <- basename( brainImageFiles[i] )
  subjectDirectory <- dirname( brainImageFiles[i] )
  subjectId <- sub( "BrainSegmentationPosteriors2.nii.gz", '', subjectId )

  brainMaskFile <- paste0( subjectDirectory, "/", subjectId, "BrainExtractionMask.nii.gz" )

  fwdtransforms <- c()
  invtransforms <- c()

  xfrmPrefix <- paste0( subjectId, "xKirbyTemplate" )
  xfrmFiles <- list.files( subjectDirectory, pattern = paste0( xfrmPrefix, "*" ), full.names = TRUE )

  fwdtransforms[1] <- xfrmFiles[3]                    # InverseWarp
  fwdtransforms[2] <- xfrmFiles[1]                    # Affine

  invtransforms[1] <- xfrmFiles[1]                    # Affine
  invtransforms[2] <- xfrmFiles[2]                    # Warp


  missingFile <- FALSE
  for( j in seq_len( length( fwdtransforms ) ) )
    {
    if( !file.exists( invtransforms[j] ) || !file.exists( fwdtransforms[j] ) )
      {
      # stop( paste( "Transform file does not exist.\n" ) )
      missingFile <- TRUE
      }
    }

  if( ! file.exists( brainImageFiles[i] ) || ! file.exists( brainMaskFile ) )
    {
    # stop( paste( "Transform file does not exist.\n" ) )
    missingFile <- TRUE
    }

  if( missingFile )
    {
    missingFiles <- append( missingFiles, subjectDirectory )
    } else {
    trainingTransforms[[count]] <- list(
      fwdtransforms = fwdtransforms, invtransforms = invtransforms )

    trainingImageFiles[[count]] <- brainImageFiles[i]
    trainingMaskFiles[[count]] <- brainMaskFile
    count <- count + 1
    }
  }
cat( "\n" )

###
#
# Create the ResNet model
#

resampledImageSize <- dim( reorientTemplate )

resnetModel <- createResNetModel3D(
  inputImageSize = c( resampledImageSize, channelSize ),
  numberOfClassificationLabels = 2,
  mode = "regression" )

resnetModel %>% compile( loss = 'mae',
  optimizer = optimizer_adam( lr = 0.0001 ),
  metrics = list( "mean_absolute_error" ) )

resnetModelWeightsFile <- paste0( scriptsDirectory, "/resnetModelWeights.h5" )

if( file.exists( resnetModelWeightsFile ) )
  {
  cat( "Loading model weights.", "\n" )
  load_model_weights_hdf5( resnetModel, resnetModelWeightsFile )
  }

###
#
# Set up the training generator
#

# Split trainingData into "training" and "validation" componets for
# training the model.

numberOfData <- length( trainingImageFiles )
sampleIndices <- sample( numberOfData )

validationSplit <- floor( 0.8 * numberOfData )
trainingIndices <- sampleIndices[1:validationSplit]
numberOfTrainingData <- length( trainingIndices )
validationIndices <- sampleIndices[( validationSplit + 1 ):numberOfData]
numberOfValidationData <- length( validationIndices )

###
#
# Run training
#

track <- resnetModel %>% fit_generator(
  generator = brainAgeBatchGenerator( batchSize = batchSize,
                                       resampledImageSize = resampledImageSize,
                                       reorientImage = reorientTemplate,
                                       sourceRegressors = regressors[trainingIndices,],
                                       sourceImageList = trainingImageFiles[trainingIndices],
                                       sourceTransformList = trainingTransforms[trainingIndices],
                                       outputFile = paste0( scriptsDirectory, "trainingData.csv" )
                                     ),
  steps_per_epoch = 32L,
  epochs = 75,
  validation_data = brainAgeBatchGenerator( batchSize = batchSize,
                                       resampledImageSize = resampledImageSize,
                                       reorientImage = reorientTemplate,
                                       sourceRegressors = regressors[validationIndices,],
                                       sourceImageList = trainingImageFiles[validationIndices],
                                       sourceTransformList = trainingTransforms[validationIndices],
                                       outputFile = paste0( scriptsDirectory, "validationData.csv" )
                                     ),
  validation_steps = 16,
  callbacks = list(
    callback_model_checkpoint( paste0( scriptsDirectory, "/resnetModelWeights.h5" ),
      monitor = 'val_loss', save_best_only = TRUE, save_weights_only = TRUE,
      verbose = 1, mode = 'auto', period = 1 ),
     callback_reduce_lr_on_plateau( monitor = 'val_loss', factor = 0.5,
       verbose = 1, patience = 10, mode = 'auto' ),
     callback_early_stopping( monitor = 'val_loss', min_delta = 0.001,
       patience = 20 )
  )
)

save_model_weights_hdf5( resnetModel, paste0( scriptsDirectory, "/resnetModelWeights.h5" ) )
