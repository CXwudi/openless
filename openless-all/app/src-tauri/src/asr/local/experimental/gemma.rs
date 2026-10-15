use super::*;
use std::ffi::{c_char, c_void, CStr, CString};
unsafe extern "C" {
    fn openless_gemma_load(path: *const c_char, error: *mut *mut c_char) -> *mut c_void;
    fn openless_gemma_transcribe(
        handle: *mut c_void,
        samples: *const f32,
        count: usize,
        error: *mut *mut c_char,
    ) -> *mut c_char;
    fn openless_gemma_release(handle: *mut c_void);
}
pub struct Handle(*mut c_void);
// Always accessed under the runtime slot mutex, including deallocation.
unsafe impl Send for Handle {}
unsafe fn take_string(value: *mut c_char) -> String {
    if value.is_null() {
        return "Unknown Core ML error".into();
    }
    let text = CStr::from_ptr(value).to_string_lossy().into_owned();
    libc::free(value.cast());
    text
}
impl Handle {
    pub fn load(path: &std::path::Path) -> Result<Self, BackendError> {
        let path = CString::new(path.to_string_lossy().as_bytes()).map_err(failure)?;
        let mut error = std::ptr::null_mut();
        let handle = unsafe { openless_gemma_load(path.as_ptr(), &mut error) };
        if handle.is_null() {
            return Err(failure(unsafe { take_string(error) }));
        }
        Ok(Self(handle))
    }
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String, BackendError> {
        let mut error = std::ptr::null_mut();
        let text = unsafe {
            openless_gemma_transcribe(self.0, samples.as_ptr(), samples.len(), &mut error)
        };
        if text.is_null() {
            return Err(failure(unsafe { take_string(error) }));
        }
        Ok(unsafe { take_string(text) })
    }
}
impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { openless_gemma_release(self.0) }
    }
}
