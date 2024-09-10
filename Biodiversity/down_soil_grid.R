rm(list = ls())
library(terra) # 引入terra包，因为使用了rast和writeRaster等函数
library(glue) # 引入glue包，用于构造字符串
library(foreach) # 引入foreach包

resolution <- "1000m" # 或者 "1000m" 或 "5000m"
output_folder <- "G:/PHD_Project/Data/SoilGrid_v2.0/" # 输出文件夹路径
quantile <- "mean" # 可选：mean, Q0.05, Q0.5, Q0.95 或 uncertainty

rst_crs <- "ESRI:54052" # nolint
# in_crs <- "EPSG:4326" # 输入的坐标参考系统
# xmin <- -180 # 边界框xmin
# xmax <- 180 # 边界框xmax
# ymin <- -90 # 边界框ymin
# ymax <- 90 # 边界框ymax
voilist <- c("wv0010", "wv0033", "wv1500", "sand", "cec", "bdod", "cfvo", "soc", "ocd", "nitrogen", "clay", "ocs", "phh2o", "silt")
depthlist <- c("0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm")

# bb <- ext(c(xmin, xmax, ymin, ymax))
for (voi in voilist) {
    for (depth in depthlist) {
        voi_layer <- paste(voi, depth, quantile, sep = "_")
        if (resolution == "250m") {
            rstFile <- paste0("/vsicurl/https://files.isric.org/soilgrids/latest/data/", voi, "/", voi_layer, ".vrt")
        } else if (resolution == "1000m") {
            rstFile <- paste0("/vsicurl/https://files.isric.org/soilgrids/latest/data_aggregated/1000m/", voi, "/", voi_layer, "_1000.tif")
        } else {
            rstFile <- paste0("/vsicurl/https://files.isric.org/soilgrids/latest/data_aggregated/5000m/", voi, "/", voi_layer, "_5000.tif")
        }
        rst <- rast(rstFile)
        crs(rst) <- rst_crs
        # bb_proj <- project(bb, crs = in_crs, new_crs = rst_crs)
        # rst <- crop(rst, bb_proj)
        output_filename <- glue("{voi}_{resolution}_{depth}.tif")
        oudir <- file.path(output_folder, voi)
        dir.create(oudir, recursive = TRUE, showWarnings = FALSE)
        outfile <- file.path(oudir, output_filename)
        writeRaster(rst, outfile, overwrite = TRUE)
        cat(outfile, "\n")
    }
}