Map.centerObject(geometry,8)

function maskS2clouds(image) {
  var qa = image.select('QA60');
  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000);
}
var rgbVis = {
  min: 0.0,
  max: 0.3,
  bands: ['B8', 'B4', 'B3'],
};
//export data
var exportdataset =  ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
                  .filterBounds(geometry)
                  .filterDate('2021-04-01', '2021-05-01')
                  // Pre-filter to get less cloudy granules.
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                  .map(maskS2clouds)
                  .select(['B8','B4','B3','B2']);
//var exportdataset2 =  ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
  //                .filterBounds(geometry)
    //              .filterDate('2017-08-01', '2017-09-01')
                  // Pre-filter to get less cloudy granules.
      //            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        //          .map(maskS2clouds)
          //        .select(['B8','B4','B3','B2']);
//var exportdataset =exportdataset1.merge(exportdataset2);

print("研究范围内的影像1",exportdataset)

  var median = exportdataset.median().clip(geometry);
  print("研究范围内的影像",median)
  Map.addLayer(median,rgbVis, 'RGB1');
  Export.image.toDrive({
      image:median,
      description:'002104',
      scale:10,
      maxPixels: 1e13,
      region:geometry,
      fileFormat: 'GeoTIFF',
      formatOptions: {
        cloudOptimized: true
      }
    })
function exportImageCollection(imgCol) {
  var indexList = imgCol.reduceColumns(ee.Reducer.toList(), ["system:index"])
                        .get("list");
                        
  print(indexList)
}

exportImageCollection(exportdataset);

