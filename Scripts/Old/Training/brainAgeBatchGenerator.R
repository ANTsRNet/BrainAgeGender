brainAgeBatchGenerator <- function( batchSize = 32L,
                                    resampledImageSize = c( 64, 64, 64 ),
                                    reorientImage = NULL,
                                    sourceRegressors = NULL,
                                    sourceImageList = NULL,
                                    sourceTransformList = NULL,
                                    outputFile = NULL )
{

  if( is.null( sourceImageList ) )
    {
    stop( "Input images must be specified." )
    }
  if( is.null( sourceRegressors ) )
    {
    stop( "Input regressors must be specified." )
    }
  if( is.null( sourceTransformList ) )
    {
    stop( "Input transforms must be specified." )
    }
  if( is.null( reorientImage ) )
    {
    stop( "No reference image specified." )
    }

  if( ! is.null( outputFile ) )
    {
    cat( "CurrentPassCount,BatchCount,Source,Reference\n", file = outputFile )
    }

  referenceImageList <- sourceImageList
  referenceTransformList <- sourceTransformList

  pairwiseIndices <- expand.grid( source = 1:length( sourceImageList ),
    reference = 1:length( referenceImageList ) )

  # shuffle the pairs
  pairwiseIndices <-
    pairwiseIndices[sample.int( nrow( pairwiseIndices ) ),]

  # shuffle the source data
  sampleIndices <- sample( length( sourceImageList ) )
  sourceImageList <- sourceImageList[sampleIndices]
  sourceTransformList <- sourceTransformList[sampleIndices]
  sourceRegressors <- sourceRegressors[sampleIndices,]

  # shuffle the reference data
  sampleIndices <- sample( length( referenceImageList ) )
  referenceImageList <- referenceImageList[sampleIndices]
  referenceTransformList <- referenceTransformList[sampleIndices]

  currentPassCount <- 0L

  function()
    {
    # Shuffle the data after each complete pass

    if( ( currentPassCount + batchSize ) >= nrow( pairwiseIndices ) )
      {
      # shuffle the source data
      sampleIndices <- sample( length( sourceImageList ) )
      sourceImageList <- sourceImageList[sampleIndices]
      sourceTransformList <- sourceTransformList[sampleIndices]
      sourceRegressors <- sourceRegressors[sampleIndices,]

      # shuffle the reference data
      sampleIndices <- sample( length( referenceImageList ) )
      referenceImageList <- referenceImageList[sampleIndices]
      referenceTransformList <- referenceTransformList[sampleIndices]

      # shuffle the pairs
      pairwiseIndices <-
        pairwiseIndices[sample.int( nrow( pairwiseIndices ) ),]

      currentPassCount <- 0L
      }

    rowIndices <- currentPassCount + 1L:batchSize

    batchIndices <- pairwiseIndices[rowIndices,]

    batchSourceImages <- sourceImageList[batchIndices$source]
    batchTransforms <- sourceTransformList[batchIndices$source]
    batchRegressors <- sourceRegressors[batchIndices$source,]

    batchReferenceImages <- referenceImageList[batchIndices$reference]
    batchReferenceTransforms <- referenceTransformList[batchIndices$reference]

    channelSize <- length( batchSourceImages[[1]] )

    batchX <- array( data = 0, dim = c( batchSize, resampledImageSize, channelSize ) )
    batchY <- array( data = 0, dim = c( batchSize, resampledImageSize ) )

    currentPassCount <<- currentPassCount + batchSize

    pb <- txtProgressBar( min = 0, max = batchSize, style = 3 )
    for( i in seq_len( batchSize ) )
      {
      setTxtProgressBar( pb, i )

      sourceChannelImages <- batchSourceImages[[i]]

      referenceXfrm <- batchReferenceTransforms[[i]]
      sourceXfrm <- batchTransforms[[i]]

      referenceChannelImageFileName <- batchReferenceImages[[i]][1]
      referenceSubjectDir <- dirname( batchReferenceImages[[i]][1] )
      referenceChannelImage <- antsImageRead( referenceChannelImageFileName, dimension = 3 )

      if( !is.null( outputFile ) )
        {
        cat( currentPassCount - batchSize, i, batchSourceImages[[i]][1], referenceChannelImageFileName,
          file = outputFile, sep = ",", append = TRUE )
        cat( "\n", file = outputFile, append = TRUE )
        }

      boolInvert <- c( TRUE, FALSE, FALSE, TRUE, FALSE )
      transforms <- c(
        referenceXfrm$fwdtransforms[2],
        referenceXfrm$fwdtransforms[1],
        referenceXfrm$fwdtransforms[2],
        sourceXfrm$invtransforms[1],
        sourceXfrm$invtransforms[2]
        )

      for( j in seq_len( channelSize ) )
        {
        sourceX <- antsImageRead( sourceChannelImages[j], dimension = 3 )

        channelTransforms <- transforms

        # cat( currentPassCount - batchSize, i, sourceChannelImages[j], referenceChannelImageFileName )
        # cat( channelTransforms, "\n" )

        warpedImageX <- antsApplyTransforms( reorientImage, sourceX,
          interpolator = "linear", transformlist = channelTransforms,
          whichtoinvert = boolInvert )

        if( any( dim( warpedImageX ) != resampledImageSize ) )
          {
          warpedArrayX <- as.array( resampleImage( warpedImageX,
            resampledImageSize, useVoxels = TRUE, interpType = 0 ) )
          } else {
          warpedArrayX <- as.array( warpedImageX )
          }

#        warpedArrayX <- ( warpedArrayX - mean( warpedArrayX ) ) /
#          sd( warpedArrayX )

        # antsImageWrite( warpedImageX, paste0( "arrayX_", i, ".nii.gz" ) )
        # readline( prompt = "Press [enter] to continue\n" )

        batchX[i,,,,j] <- warpedArrayX
        }
      }
    cat( "\n" )

    return( list( batchX, batchRegressors ) )
    }
}
