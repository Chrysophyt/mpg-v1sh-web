import Head from 'next/head'
import Image from 'next/image'
import styles from '../../styles/Home.module.css'


import * as ort from 'onnxruntime-web';
import { Tensor } from 'onnxruntime-web';
import { useState } from 'react'
import { Chart } from 'chart.js';

const img_h = 300;
const img_w = 500;

const K=12
const pixelSize = 10
const pixelPad = 5
const pixelRadius = pixelSize//2
var orientations = new Array()
for (let k=0; k<K; k++){
  orientations.push((Math.PI*k/K) - (Math.PI/2))
}


var feedsCompute= {};
var sessionCompute = 0

var canvasOutput = 0
var iterations = 1


//Showing statistics
// var chartavg = new Chart(avgSpCanvas, {
//   type: 'line',
//   data: {
//     labels:[0,1,2,3],
//     datasets: [{
//       label: 'Average Excitatory',
//       data: [65, 59, 80, 81, 56, 55, 40],
//       fill: false,
//       borderColor: 'rgb(75, 192, 192)',
//     }]
//   }
//   })
var avgSpChart
var avgSmChart

var avgOutSpChart
var avgOutSmChart


const drawOutput = async (event) => {
  const start = new Date()

  const outputDataCompute  = await sessionCompute.run(feedsCompute)
  feedsCompute[sessionCompute.inputNames[1]] = outputDataCompute[sessionCompute.outputNames[0]]
  feedsCompute[sessionCompute.inputNames[2]] = outputDataCompute[sessionCompute.outputNames[1]]

  
  const end = new Date()
  const inferenceTime = (end.getTime() - start.getTime())

  // 'avgSp', 'avgSm', 'avgOutSp', 'avgOutSm'
  addData(avgSpChart, iterations, outputDataCompute[sessionCompute.outputNames[4]].data)
  addData(avgSmChart, iterations, outputDataCompute[sessionCompute.outputNames[5]].data)
  addData(avgOutSpChart, iterations, outputDataCompute[sessionCompute.outputNames[6]].data)
  addData(avgOutSmChart, iterations, outputDataCompute[sessionCompute.outputNames[7]].data)

  drawHypercolumns(canvasOutput, outputDataCompute[sessionCompute.outputNames[2]].data, 56, 96, true)
  document.getElementById('inference_time').innerHTML = inferenceTime
  document.getElementById('iterations').innerHTML = iterations

  iterations += 1
  window.requestAnimationFrame(drawOutput)

}

function addData(chart, label, data) {
  chart.data.labels.push(label);
  chart.data.datasets[0].data.push(data);
  chart.update();
}

function drawLine(ctx, strength, orient, radius, start_point, end_point){
  
  const mid_point_x = ((start_point[0]+end_point[0])/2)
  const mid_point_y = ((start_point[1]+end_point[1])/2)

  const new_point1_x = (mid_point_x + radius * Math.cos(orient))
  const new_point1_y = (mid_point_y - radius * Math.sin(orient)) 

  const new_point2_x = (mid_point_x + radius * Math.cos(orient+Math.PI))
  const new_point2_y = (mid_point_y - radius * Math.sin(orient+Math.PI))


  ctx.beginPath()
  ctx.moveTo(Math.round(new_point1_x), Math.round(new_point1_y))
  ctx.lineTo(Math.round(new_point2_x), Math.round(new_point2_y))
  ctx.lineWidth = strength
  // ctx.strokeStyle = '#000000'
  ctx.stroke()
}

function drawHypercolumns(canvas, hypercolumns, height, width, isOutput){
  var ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let k=0; k<K; k++){
    for (let h=0; h<height; h++){
      for (let w=0; w<width; w++){
        if(hypercolumns[(k*height*width)+h*width+w]<=0){
          continue
        }

        var strength = hypercolumns[(k*height*width)+h*width+w]
        if (isOutput == true){
          strength = strength * 9
        } 

        drawLine(ctx, strength, orientations[k], pixelRadius, 
          [w * (pixelSize + 2*pixelPad), h * (pixelSize + 2*pixelPad)], 
          [(w+1)*(pixelSize + 2*pixelPad), (h+1) * (pixelSize + 2*pixelPad)])
      }
    }
  }
}  


export default function Home() {
  const [img, setImg] = useState()
  const [canvasState, setCanvas] = useState()
  const [outputCanvasState, setOutputCanvas] = useState()





  const inputData = async (event) => {
    event.preventDefault();
    const imageFile = document.getElementById('image').files[0]
    if(imageFile==null){
      return
    }
    // show canvas
    await setCanvas(1)
    await setOutputCanvas(1)

    var hcanvas = document.getElementById("myCanvas");

    // resize to designated res 300x500
    const canvas = document.createElement('canvas')
    
    // scale & draw the image onto the canvas
    const ctx = canvas.getContext('2d')

    canvas.width = img_w
    canvas.height = img_h
    var img = await createImageBitmap(imageFile);
    ctx.drawImage(img, 0, 0, img_w, img_h)
    
    // Get the binary (aka blob)
    const blob = await new Promise(rs => canvas.toBlob(rs, 1))
    const resizedFile = new File([blob], imageFile.name, imageFile)

    //set image
    setImg(URL.createObjectURL(resizedFile));

    //to float32
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imgData.data
    const float32Data = new Float32Array(canvas.width*canvas.height)
    
    var data_iter = 0
    for(let i = 0; i < data.length; i += 4) {
      float32Data[data_iter] = (data[i]*0.299)/255.0 + (data[i+1]*0.587)/255.0 + (data[i+2]*0.114)/255.0
      data_iter +=1
    }
    //const transposedData = rArray.concat(gArray).concat(bArray);

    const inputTensor = new Tensor("float32", float32Data, [300, 500]);

    //const inputTensor = new Tensor("float32", float32Data, [canvas.width, canvas.height]);

    

    // INITIALIZING MODEL
    const session = await ort.InferenceSession
                          .create('/GTHModel.onnx',
                          { executionProviders: ['webgl'], graphOptimizationLevel: 'all' });
    //console.log('Inference session created');

    //const start = new Date();
    //const feeds: Record<string, ort.Tensor> = {};
    const feeds = {};
    feeds[session.inputNames[0]] = inputTensor
    var outputData  = await session.run(feeds);
    var outputData  = outputData[session.outputNames[0]]

    //const end = new Date();
    //const inferenceTime = (end.getTime() - start.getTime())/1000;
    
    hcanvas.height = 56 * (pixelSize+2*pixelPad)
    hcanvas.width = 96 * (pixelSize+2*pixelPad)
    await drawHypercolumns(hcanvas, outputData.data, 56, 96, false)

    const sessionInit = await ort.InferenceSession
                          .create('/V1SHNetInitModel.onnx',
                          { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    console.log('Inference session created');


    const feedsInit = {};
    var outputDataInit  = await sessionInit.run(feedsInit);


    var Sp = outputDataInit[sessionInit.outputNames[0]]
    var Sm = outputDataInit[sessionInit.outputNames[1]]

    //Create output canvas
    canvasOutput = document.getElementById("outputCanvas");
    canvasOutput.height = 56 * (pixelSize+2*pixelPad)
    canvasOutput.width = 96 * (pixelSize+2*pixelPad)
    
    
    sessionCompute = await ort.InferenceSession
                          .create('/V1SHComputeMachineModel.onnx',
                          { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });

    feedsCompute[sessionCompute.inputNames[0]] = outputData
    
    feedsCompute[sessionCompute.inputNames[1]] = Sp
    feedsCompute[sessionCompute.inputNames[2]] = Sm

    
    const avgSpCanvas = document.getElementById('avgSpCanvas')
    const avgSmCanvas = document.getElementById('avgSmCanvas')
    const avgOutSpCanvas = document.getElementById('avgOutSpCanvas')
    const avgOutSmCanvas = document.getElementById('avgOutSmCanvas')

    // setup charts
    avgSpChart = new Chart(avgSpCanvas, {
        type: 'line',
        data: {
          labels:[0],
          datasets: [{
            label: 'Avg Sp',
            data: [0],
            fill: false,
            borderColor: 'rgb(250, 110, 110)',
          }]
        },
        options: {
          datasets: {
              line: {
                  pointRadius: 0 // disable for all `'line'` datasets
               }
          },
          scales: {
            yAxes: [{
              ticks: {
                suggestedMin: 0,
                suggestedMax: 0.2
              }
            }]
          }
        },
        
    })

    avgSmChart = new Chart(avgSmCanvas, {
      type: 'line',
      data: {
        labels:[0],
        datasets: [{
          label: 'Avg Sm',
          data: [0],
          fill: false,
          borderColor: 'rgb(75, 190, 190)',
        }]
      },
      options: {
        datasets: {
            line: {
                pointRadius: 0 // disable for all `'line'` datasets
             }
        },
        scales: {
          yAxes: [{
            ticks: {

              suggestedMin: 0,
              suggestedMax: 1.5

            }
          }]
        }
      }
      })

    avgOutSpChart = new Chart(avgOutSpCanvas, {
      type: 'line',
      data: {
        labels:[0],
        datasets: [{
          label: 'Avg Sp Out',
          data: [0],
          fill: false,
          borderColor: 'rgb(250, 160, 160)',
        }]
      },
      options: {
        datasets: {
            line: {
                pointRadius: 0 // disable for all `'line'` datasets
             }
        },
        scales: {
          yAxes: [{
            ticks: {
              suggestedMin: 0,
              suggestedMax: 0.01
            }
          }]
        }
      }
      })

      avgOutSmChart = new Chart(avgOutSmCanvas, {
        type: 'line',
        data: {
          labels:[0],
          datasets: [{
            label: 'Avg Sm Out',
            data: [0],
            fill: false,
            borderColor: 'rgb(150, 220, 220)',
          }]
        },
        options: {
          datasets: {
              line: {
                  pointRadius: 0 // disable for all `'line'` datasets
               }
          },
          scales: {
            yAxes: [{
              ticks: {
                scaleOverride : true,
                suggestedMin: 0,
                suggestedMax: 0.5
              }
            }]
          }
        }
      })


    // START LOOP
    window.requestAnimationFrame(drawOutput)
    // for (let i=0;i<100;i++){
    
    //   const start = new Date();
    //   feedsCompute[sessionCompute.inputNames[1]] = outputDataCompute[sessionCompute.outputNames[0]]
    //   feedsCompute[sessionCompute.inputNames[2]] = outputDataCompute[sessionCompute.outputNames[1]]
    //   feedsCompute[sessionCompute.inputNames[3]] = outputDataCompute[sessionCompute.outputNames[2]]
    //   feedsCompute[sessionCompute.inputNames[4]] = outputDataCompute[sessionCompute.outputNames[3]]
    //   outputDataCompute  = await sessionCompute.run(feedsCompute)
    //   const end = new Date()
    //   const inferenceTime = (end.getTime() - start.getTime())/1000

    //   canvasData = await feedsCompute[sessionCompute.inputNames[1]].data
    //   console.log(inferenceTime)
    // }
    

    // show canvas
    




  }



  return (
    <div className={styles.container}>
        <Head>
        <title>Contour Model | MPI for Biological Cybernetics</title>
        <meta name="description" content="Generated by create next app" />
        <link rel="icon" href="/icon.png" />
        <link
          href="https://fonts.googleapis.com/css2?family=Roboto+Mono&display=optional"
          rel="stylesheet"
        />
        </Head>

      <main className={styles.main}>
        <h2 className={styles.description} >
          A Neural Model of Contour Integration in the Primary Visual Cortex
          
        </h2>
        <h3>Zhaoping Li</h3>
        

      <div className={styles.grid}>
        <div  className={styles.card}>
        <h3>Documentation &rarr;</h3>
            <div>This is a web implementation of <i>A Neural Model of Contour Integration in the Primary Visual Cortex</i> <a href="https://pubmed.ncbi.nlm.nih.gov/9573412/">paper</a>.</div>
        </div>

        <div className={styles.card}>
          <h3>Input image</h3>

          <form onSubmit={inputData}>
            <label form="inputData">Input Image Data to Load:</label><br></br>
            <input type="file" id="image" />
            <button type="submit">Compute</button>
          </form>

          
          <div className="images" >
          {img && <Image 
          src={img} 
          alt="" 
          width={img_w}
          height={img_h}    
          />}
          </div>
        </div>

        {canvasState && 
        <div className={styles.cardLarge}>
          <h3>Input Hypercolumns</h3>
          <canvas className={styles.canvas}
          id="myCanvas" 
          width="500" 
          height="300"/>
        </div>}

        {outputCanvasState && 
        <div className={styles.cardLarge}>
          <h3>Excitatory Gain Output</h3>
          <div className={styles.textstat} >Inference time: <div className={styles.textstat} id="inference_time"></div>ms</div>
          <br></br>
          <div className={styles.textstat} >Current iterations: <div className={styles.textstat} id="iterations"></div></div>

          <canvas className={styles.canvas}
          id="outputCanvas" 
          width="500" height="300"></canvas>
        </div>}

        {outputCanvasState && 
        <div  className={styles.card}>
        <h3>Average Excitatory</h3>
            <canvas className={styles.canvas}
          id="avgSpCanvas" 
          width="400" height="400"></canvas>
        </div>}

        {outputCanvasState && 
        <div  className={styles.card}>
        <h3>Average Inhibitory</h3>
            <canvas className={styles.canvas}
          id="avgSmCanvas" 
          width="400" height="400"></canvas>
        </div>}

        {outputCanvasState && 
        <div  className={styles.card}>
        <h3>Average Excitatory Out</h3>
            <canvas className={styles.canvas}
          id="avgOutSpCanvas" 
          width="400" height="400"></canvas>
        </div>}

        {outputCanvasState && 
        <div  className={styles.card}>
        <h3>Average Inhibitory Out</h3>
            <canvas className={styles.canvas}
          id="avgOutSmCanvas" 
          width="400" height="400"></canvas>
        </div>}

      </div>



      </main>

      <footer className={styles.footer}>
        <a
          href="https://www.kyb.tuebingen.mpg.de/"
          target="_blank"
          rel="noopener noreferrer"
        >
        <Image src="/kybmpg.svg" height={150} width={400} />
        </a>
        
        <div>
          Computation and Cognition Tübingen Summer 2022
        </div>
      </footer>
    </div>
  )
}

