
export const byId = (id) => document.getElementById(id)
export const onClick = (element, func) => element.addEventListener('click', func)
export const toggleAllButtons = (toggle) => document.querySelectorAll('.button').forEach(b => b.setAttribute('disabled', !toggle))
export const setStatus = (status = 'loaded') => document.body.setAttribute('status', status)

export async function initiateWebcam(videoTag) {
  return new Promise(async (resolve, reject) => {
    const navigatorAny = navigator
    navigator.getUserMedia = navigator.getUserMedia ||
        navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
        navigatorAny.msGetUserMedia

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        videoTag.srcObject = stream
        videoTag.addEventListener('loadeddata',  () => resolve(), false)
      }
      catch(e) {
        reject()
      }
    }
    else if (navigator.getUserMedia) {
      navigator.getUserMedia({ video: true, audio: false },
        stream => {
          videoTag.srcObject = stream
          videoTag.addEventListener('loadeddata',  () => resolve(), false)
        },
        error => reject())
    } else {
      reject()
    }
  })
}

export const normalizeAudio = (a, mean = -100, std = 100) => a.map(x => (x - mean) / std)
export const flattenAudioSamples = tensors => {
 const size = tensors[0].length
 const result = new Float32Array(tensors.length * size)
 tensors.forEach((arr, i) => result.set(arr, i * size))

 return result
}

export const throttle = (fn, wait) => {
  let previouslyRun, queuedToRun;

  return function invokeFn(...args) {
    const now = Date.now();

    queuedToRun = clearTimeout(queuedToRun);

    if (!previouslyRun || (now - previouslyRun >= wait)) {
      fn.apply(null, args);
      previouslyRun = now;
    } else {
      queuedToRun = setTimeout(invokeFn.bind(null, ...args), wait - (now - previouslyRun));    
    }
  }
}; 
