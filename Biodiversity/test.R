library(BIEN)
library(ape) #Package for working with phylogenies in R
library(maps) #Useful for making quick maps of occurrences

library(sf)
vignette("BIEN")
Xanthium_strumarium <- BIEN_occurrence_species(species = "Xanthium strumarium")
str(Xanthium_strumarium)
head(Xanthium_strumarium)
LUQUILLO_full <- BIEN_plot_name(plot.name = "LUQUILLO",
                                cultivated = TRUE,
                                all.taxonomy = TRUE,
                                native.status = TRUE,
                                political.boundaries = TRUE,
                                all.metadata = TRUE)
library(rgbif)
help.start()
library(ALA4R)
occurrences(taxon="data_resource_uid:dr356",record_count_only=TRUE)
dataset_doi('10.15468/igasai')
ala_config()
layers = c('cl22','cl23','el773')
pnts = c(-23.1,149.1)
intersect_points(pnts,layers)

885.34+2278.8
2278.8

install.packages('FD')
library(FD)
head(dummy)
aa <- dbFD(dummy$trait, dummy$abun)
summary(aa)
