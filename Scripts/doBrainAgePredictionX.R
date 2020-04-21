library( ANTsR )
library( ANTsRNet )
library( keras )
library( brainAgeR )


args <- commandArgs( trailingOnly = TRUE )

if( length( args ) == 0 )
  {
  helpMessage <- paste0( "Usage:  Rscript doBrainAgePrediction.R outputCsvFile inputT1_1 inputT1_2 inputT1_3 ...\n" )
  stop( helpMessage )
  } else {
  outputCsvFile <- args[1]
  inputFileNames <- args[2:length( args )]
  }

#################
#
#  Main routine
#

verbose <- TRUE

model <- getBrainAgeModel( tempfile() ) # download from figshare

brainAgesMean <- rep( NA, length( inputFileNames ) )
brainAgesStd <- rep( NA, length( inputFileNames ) )
brainGendersMean <- rep( NA, length( inputFileNames ) )
brainGendersStd <- rep( NA, length( inputFileNames ) )
for( i in seq_len( length( inputFileNames ) ) )
  {
  inputImage <- antsImageRead( inputFileNames[i] )
  if( verbose )
    {
    cat( "Preprocessing input image ", inputFileNames[i], ".\n", sep = '' )
    }

  brainAge <- brainAge( inputImage, batch_size = 10, sdAff = 0.01, model = model )

  brainAgesMean[i] <- mean( as.numeric( bage[[1]][, 1] ), na.rm = TRUE )
  brainAgesStd[i] <- sd( as.numeric( bage[[1]][, 1] ), na.rm = TRUE )
  brainGendersMean[i] <- mean( as.numeric( bage[[1]][, 2] ), na.rm = TRUE )
  brainGendersStd[i] <- sd( as.numeric( bage[[1]][, 2] ), na.rm = TRUE )
  }

brainAgeDataFrame <- data.frame( FileName = inputFileNames, Age = brainAgesMean,
  Gender = brainGendersMean )

if( outputCsvFile != "None" && outputCsvFile != "none" )
  {
  write.csv( brainAgeDataFrame, file = outputCsvFile, row.names = FALSE )
  } else {
  print( brainAgeDataFrame )
  }




