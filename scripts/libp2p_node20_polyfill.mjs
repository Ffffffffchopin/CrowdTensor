if (typeof Promise.withResolvers !== 'function') {
  Object.defineProperty(Promise, 'withResolvers', {
    configurable: true,
    writable: true,
    value: () => {
      let resolve
      let reject
      const promise = new Promise((innerResolve, innerReject) => {
        resolve = innerResolve
        reject = innerReject
      })
      return { promise, resolve, reject }
    }
  })
}
