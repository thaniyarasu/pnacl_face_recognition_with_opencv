// Copyright 2014 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <string.h>

#include <vector>

#include <pthread.h>
#include <stdio.h>

// #include <iostream>       // std::cout
// #include <thread>         // std::thread, std::thread::id, std::this_thread::get_id
// #include <chrono>
// #include <stdio.h>
// #include <pthread.h>
// #include <sys/types.h>
// #include <unistd.h>

#include "ppapi/c/pp_errors.h"
#include "ppapi/c/ppb_opengles2.h"
#include "ppapi/cpp/completion_callback.h"
#include "ppapi/cpp/graphics_3d.h"
#include "ppapi/cpp/graphics_3d_client.h"
#include "ppapi/cpp/instance.h"
#include "ppapi/cpp/media_stream_video_track.h"
#include "ppapi/cpp/module.h"
#include "ppapi/cpp/rect.h"
#include "ppapi/cpp/var.h"
#include "ppapi/cpp/var_dictionary.h"
#include "ppapi/cpp/video_frame.h"
#include "ppapi/lib/gl/gles2/gl2ext_ppapi.h"
#include "ppapi/utility/completion_callback_factory.h"

#include "ppapi/cpp/url_loader.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "ppapi/cpp/url_loader.h"
#include "ppapi/cpp/url_request_info.h"
#include "ppapi/cpp/url_response_info.h"

// When compiling natively on Windows, PostMessage can be #define-d to
// something else.
#ifdef PostMessage
#undef PostMessage
#endif

// Assert |context_| isn't holding any GL Errors.  Done as a macro instead of a
// function to preserve line number information in the failure message.
#define AssertNoGLError() \
  PP_DCHECK(!glGetError());

namespace {
//const char* const
  const int kBufSize = 1024*128;

  const char* const FaceCascadeName = "haarcascades/haarcascade_frontalface_alt.xml";
  //const char* const EyeCascadeName = "haarcascade_eye_tree_eyeglasses.xml";

// This object is the global object representing this plugin library as long
// as it is loaded.
class MediaStreamVideoModule : public pp::Module {
 public:
  MediaStreamVideoModule() : pp::Module() {}
  virtual ~MediaStreamVideoModule() {}

  virtual pp::Instance* CreateInstance(PP_Instance instance);
};

class MediaStreamVideoDemoInstance : public pp::Instance,
                        public pp::Graphics3DClient {
 public:
  MediaStreamVideoDemoInstance(PP_Instance instance, pp::Module* module);
  virtual ~MediaStreamVideoDemoInstance();

  // pp::Instance implementation (see PPP_Instance).
  virtual void DidChangeView(const pp::Rect& position,
                             const pp::Rect& clip_ignored);
  virtual void HandleMessage(const pp::Var& message_data);


  // pp::Graphics3DClient implementation.
  virtual void Graphics3DContextLost() {
    //cv::Mat frame;
    //cv::CascadeClassifier frontal_cascade;
    //cv::VideoCapture capture;

    InitGL();
    CreateTextures();
    Render();
  }

 private:
  void DrawYUV();
  void DrawRGB();
  void Render();

  // GL-related functions.
  void InitGL();
  GLuint CreateTexture(int32_t width, int32_t height, int unit, bool rgba);
  void CreateGLObjects();
  void CreateShader(GLuint program, GLenum type, const char* source);
  void PaintFinished(int32_t result);
  void CreateTextures();
  void ConfigureTrack();

  void Start();
  void Opened(int32_t);
  void Read();
  void Readed(int32_t);
  void ReportError(const std::string&);
  void CreateFaceCascade(const std::string& );
  void logmsg(const char*);
  void PrintThreadId(pthread_t id);


  void errormsg(const char*);
  void DetectAndDisplay();
  // MediaStreamVideoTrack callbacks.
  void OnConfigure(int32_t result);
  void OnGetFrame(int32_t result, pp::VideoFrame frame);

  pp::Size position_size_;
  bool is_painting_;
  bool needs_paint_;
  bool is_bgra_;
  GLuint program_yuv_;
  GLuint program_rgb_;
  GLuint buffer_;
  GLuint texture_y_;
  GLuint texture_u_;
  GLuint texture_v_;
  GLuint texture_rgb_;
  pp::MediaStreamVideoTrack video_track_;
  pp::CompletionCallbackFactory<MediaStreamVideoDemoInstance> callback_factory_;
  std::vector<int32_t> attrib_list_;

  // MediaStreamVideoTrack attributes:
  bool need_config_;
  PP_VideoFrame_Format attrib_format_;
  int32_t attrib_width_;
  int32_t attrib_height_;

  // Owned data.
  pp::Graphics3D* context_;

  pp::Size frame_size_;

  // added

  pp::URLLoader loader_;
  pp::URLResponseInfo response_;
  std::string content_;
  char buf_[kBufSize];

  cv::CascadeClassifier FaceCascade;
  cv::CascadeClassifier EyeCascade;
  int imageCaptured;
  cv::Mat image;
};

MediaStreamVideoDemoInstance::MediaStreamVideoDemoInstance(
    PP_Instance instance, pp::Module* module)
    : pp::Instance(instance),
      pp::Graphics3DClient(this),
      is_painting_(false),
      needs_paint_(false),
      is_bgra_(false),
      texture_y_(0),
      texture_u_(0),
      texture_v_(0),
      texture_rgb_(0),
      callback_factory_(this),
      need_config_(false),
      attrib_format_(PP_VIDEOFRAME_FORMAT_I420),
      attrib_width_(0),
      attrib_height_(0),
      context_(NULL) {
  if (!glInitializePPAPI(pp::Module::Get()->get_browser_interface())) {
    LogToConsole(PP_LOGLEVEL_ERROR, pp::Var("Unable to initialize GL PPAPI!"));
    assert(false);
  }
}

MediaStreamVideoDemoInstance::~MediaStreamVideoDemoInstance() {
  delete context_;
}

void MediaStreamVideoDemoInstance::DidChangeView(
    const pp::Rect& position, const pp::Rect& clip_ignored) {
  if (position.width() == 0 || position.height() == 0)
    return;
  if (position.size() == position_size_)
    return;

  position_size_ = position.size();

  // Initialize graphics.
  InitGL();
  Render();
}

void MediaStreamVideoDemoInstance::Start(){
  logmsg("------------------> start starting");
  content_.clear();
  pp::URLRequestInfo request(this);
  request.SetURL(FaceCascadeName);
  request.SetMethod("GET");
  loader_ = pp::URLLoader(this);
  loader_.Open(request, callback_factory_.NewCallback(&MediaStreamVideoDemoInstance::Opened));
  logmsg("------------------> stop starting");
}

void MediaStreamVideoDemoInstance::Opened(int32_t result) {
  logmsg("------------------> start opened");
  printf("%zu\n",result);
  if (result != PP_OK) {
    ReportError("URL could not be requested");
    return;
  }
  response_ = loader_.GetResponseInfo();
  Read();
  logmsg("------------------> end opened");
}  

void MediaStreamVideoDemoInstance::Read(){
  //logmsg("------------------> start read");
  pp::CompletionCallback cc = callback_factory_.NewOptionalCallback(&MediaStreamVideoDemoInstance::Readed);
  int32_t result = PP_OK;
  do {
    result = loader_.ReadResponseBody(buf_, kBufSize, cc);
    if (result > 0) content_.append(buf_, result);
  } while (result > 0);
  if (result != PP_OK_COMPLETIONPENDING) cc.Run(result);
}
  
void MediaStreamVideoDemoInstance::Readed(int32_t result) {
  if (result == PP_OK) {
    CreateFaceCascade(content_);
  } else if (result > 0) {
    content_.append(buf_, result);
    Read();
  } else {
    ReportError("A read error occurred");
  }
}

void MediaStreamVideoDemoInstance::ReportError(const std::string& data) {
  PostMessage(pp::Var(data));
}

void MediaStreamVideoDemoInstance::CreateFaceCascade(const std::string& data) {
  printf("%d\n",data.size());
  cv::FileStorage fs(data, cv::FileStorage::READ | cv::FileStorage::MEMORY);
  int i = FaceCascade.read(fs.getFirstTopLevelNode());
  printf("%d\n",i );
}
void MediaStreamVideoDemoInstance::PrintThreadId(pthread_t id){
  size_t i;
  for (i = sizeof(i); i; --i)
    printf("%02x", *(((unsigned char*) &id) + i - 1));
}

void MediaStreamVideoDemoInstance::logmsg(const char* pMsg){
  fprintf(stdout,"logmsg: %s\n",pMsg);
}

void MediaStreamVideoDemoInstance::errormsg(const char* pMsg){
  fprintf(stderr,"logerr: %s\n",pMsg);
}
void MediaStreamVideoDemoInstance::DetectAndDisplay() {
  std::vector<cv::Rect> faces;
  cv::Mat frame_gray;
  //cv::cvtColor( image, frame_gray, cv::COLOR_BGRA2GRAY );

  cv::cvtColor( image, frame_gray, cv::COLOR_YUV2GRAY_YV12 );

  cv::equalizeHist( frame_gray, frame_gray );
   //-- Detect faces
  FaceCascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );
 
  PrintThreadId(pthread_self());
  for( size_t i = 0; i < faces.size(); i++ )
  {
    logmsg("match found");
    cv::Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
    cv::ellipse( image, center, cv::Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, cv::Scalar( 255, 0, 255 ), 2, 8, 0 );
 
    // cv::Mat faceROI = frame_gray( faces[i] );
    // std::vector<Rect> eyes;

    //-- In each face, detect eyes
    // eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    // for( size_t j = 0; j < eyes.size(); j++ )
    //  {
    //    Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
    //    int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
    //    circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 3, 8, 0 );
    //  }
  }
     logmsg("end");
 //http://docs.opencv.org/modules/core/doc/basic_structures.html#mat

     //http://docs.opencv.org/modules/core/doc/drawing_functions.html
}

void MediaStreamVideoDemoInstance::HandleMessage(const pp::Var& var_message) {
logmsg(var_message.AsString().c_str());

  if (!var_message.is_dictionary()) {
    LogToConsole(PP_LOGLEVEL_ERROR, pp::Var("Invalid message!"));
    return;
  }

  pp::VarDictionary var_dictionary_message(var_message);
  std::string command = var_dictionary_message.Get("command").AsString();
  //command = "format";
  if (command == "init") {
    pp::Var var_track = var_dictionary_message.Get("track");
    if (!var_track.is_resource())
      return;
    pp::Resource resource_track = var_track.AsResource();
    video_track_ = pp::MediaStreamVideoTrack(resource_track);
    ConfigureTrack();
    //logmsg("if");
  } else if (command == "format") {
    std::string str_format = var_dictionary_message.Get("format").AsString();
    //str_format = "BGRA";
    //logmsg("else");
    //logmsg(str_format.c_str());
    if (str_format == "YV12") {
      attrib_format_ = PP_VIDEOFRAME_FORMAT_YV12;
    } else if (str_format == "I420") {
      attrib_format_ = PP_VIDEOFRAME_FORMAT_I420;
    } else if (str_format == "BGRA") {
      attrib_format_ = PP_VIDEOFRAME_FORMAT_BGRA;
    } else {
      attrib_format_ = PP_VIDEOFRAME_FORMAT_UNKNOWN;
    }
    need_config_ = true;
  } else if (command == "size") {
    attrib_width_ = var_dictionary_message.Get("width").AsInt();
    attrib_height_ = var_dictionary_message.Get("height").AsInt();
    need_config_ = true;
  } else {
    LogToConsole(PP_LOGLEVEL_ERROR, pp::Var("Invalid command!"));
  }
  Start();
}

void MediaStreamVideoDemoInstance::InitGL() {
  PP_DCHECK(position_size_.width() && position_size_.height());
  is_painting_ = false;

  delete context_;
  int32_t attributes[] = {
    PP_GRAPHICS3DATTRIB_ALPHA_SIZE, 0,
    PP_GRAPHICS3DATTRIB_BLUE_SIZE, 8,
    PP_GRAPHICS3DATTRIB_GREEN_SIZE, 8,
    PP_GRAPHICS3DATTRIB_RED_SIZE, 8,
    PP_GRAPHICS3DATTRIB_DEPTH_SIZE, 0,
    PP_GRAPHICS3DATTRIB_STENCIL_SIZE, 0,
    PP_GRAPHICS3DATTRIB_SAMPLES, 0,
    PP_GRAPHICS3DATTRIB_SAMPLE_BUFFERS, 0,
    PP_GRAPHICS3DATTRIB_WIDTH, position_size_.width(),
    PP_GRAPHICS3DATTRIB_HEIGHT, position_size_.height(),
    PP_GRAPHICS3DATTRIB_NONE,
  };
  context_ = new pp::Graphics3D(this, attributes);
  PP_DCHECK(!context_->is_null());

  glSetCurrentContextPPAPI(context_->pp_resource());

  // Set viewport window size and clear color bit.
  glClearColor(1, 0, 0, 1);
  glClear(GL_COLOR_BUFFER_BIT);
  glViewport(0, 0, position_size_.width(), position_size_.height());

  BindGraphics(*context_);
  AssertNoGLError();

  CreateGLObjects();
}

void MediaStreamVideoDemoInstance::DrawYUV() {
  static const float kColorMatrix[9] = {
    1.1643828125f, 1.1643828125f, 1.1643828125f,
    0.0f, -0.39176171875f, 2.017234375f,
    1.59602734375f, -0.81296875f, 0.0f
  };

  glUseProgram(program_yuv_);
  glUniform1i(glGetUniformLocation(program_yuv_, "y_texture"), 0);
  glUniform1i(glGetUniformLocation(program_yuv_, "u_texture"), 1);
  glUniform1i(glGetUniformLocation(program_yuv_, "v_texture"), 2);
  glUniformMatrix3fv(glGetUniformLocation(program_yuv_, "color_matrix"),
      1, GL_FALSE, kColorMatrix);
  AssertNoGLError();

  GLint pos_location = glGetAttribLocation(program_yuv_, "a_position");
  GLint tc_location = glGetAttribLocation(program_yuv_, "a_texCoord");
  AssertNoGLError();
  glEnableVertexAttribArray(pos_location);
  glVertexAttribPointer(pos_location, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(tc_location);
  glVertexAttribPointer(tc_location, 2, GL_FLOAT, GL_FALSE, 0,
      static_cast<float*>(0) + 16);  // Skip position coordinates.
  AssertNoGLError();

  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  AssertNoGLError();
}

void MediaStreamVideoDemoInstance::DrawRGB() {
  glUseProgram(program_rgb_);
  glUniform1i(glGetUniformLocation(program_rgb_, "rgb_texture"), 3);
  AssertNoGLError();

  GLint pos_location = glGetAttribLocation(program_rgb_, "a_position");
  GLint tc_location = glGetAttribLocation(program_rgb_, "a_texCoord");
  AssertNoGLError();
  glEnableVertexAttribArray(pos_location);
  glVertexAttribPointer(pos_location, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(tc_location);
  glVertexAttribPointer(tc_location, 2, GL_FLOAT, GL_FALSE, 0,
      static_cast<float*>(0) + 16);  // Skip position coordinates.
  AssertNoGLError();

  glDrawArrays(GL_TRIANGLE_STRIP, 4, 4);
}

void MediaStreamVideoDemoInstance::Render() {
  PP_DCHECK(!is_painting_);
  is_painting_ = true;
  needs_paint_ = false;

  if (texture_y_) {
    DrawRGB();
    DrawYUV();
  } else {
    glClear(GL_COLOR_BUFFER_BIT);
  }
  pp::CompletionCallback cb = callback_factory_.NewCallback(
      &MediaStreamVideoDemoInstance::PaintFinished);
  context_->SwapBuffers(cb);
}

void MediaStreamVideoDemoInstance::PaintFinished(int32_t result) {
  is_painting_ = false;
  if (needs_paint_)
    Render();
}

GLuint MediaStreamVideoDemoInstance::CreateTexture(
    int32_t width, int32_t height, int unit, bool rgba) {
  GLuint texture_id;
  glGenTextures(1, &texture_id);
  AssertNoGLError();

  // Assign parameters.
  glActiveTexture(GL_TEXTURE0 + unit);
  glBindTexture(GL_TEXTURE_2D, texture_id);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  // Allocate texture.
  glTexImage2D(GL_TEXTURE_2D, 0,
               rgba ? GL_BGRA_EXT : GL_LUMINANCE,
               width, height, 0,
               rgba ? GL_BGRA_EXT : GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);
  AssertNoGLError();
  return texture_id;
}

void MediaStreamVideoDemoInstance::CreateGLObjects() {
  // Code and constants for shader.
  static const char kVertexShader[] =
      "varying vec2 v_texCoord;            \n"
      "attribute vec4 a_position;          \n"
      "attribute vec2 a_texCoord;          \n"
      "void main()                         \n"
      "{                                   \n"
      "    v_texCoord = a_texCoord;        \n"
      "    gl_Position = a_position;       \n"
      "}";

  static const char kFragmentShaderYUV[] =
      "precision mediump float;                                   \n"
      "varying vec2 v_texCoord;                                   \n"
      "uniform sampler2D y_texture;                               \n"
      "uniform sampler2D u_texture;                               \n"
      "uniform sampler2D v_texture;                               \n"
      "uniform mat3 color_matrix;                                 \n"
      "void main()                                                \n"
      "{                                                          \n"
      "  vec3 yuv;                                                \n"
      "  yuv.x = texture2D(y_texture, v_texCoord).r;              \n"
      "  yuv.y = texture2D(u_texture, v_texCoord).r;              \n"
      "  yuv.z = texture2D(v_texture, v_texCoord).r;              \n"
      "  vec3 rgb = color_matrix * (yuv - vec3(0.0625, 0.5, 0.5));\n"
      "  gl_FragColor = vec4(rgb, 1.0);                           \n"
      "}";

  static const char kFragmentShaderRGB[] =
      "precision mediump float;                                   \n"
      "varying vec2 v_texCoord;                                   \n"
      "uniform sampler2D rgb_texture;                             \n"
      "void main()                                                \n"
      "{                                                          \n"
      "  gl_FragColor = texture2D(rgb_texture, v_texCoord);       \n"
      "}";

  // Create shader programs.
  program_yuv_ = glCreateProgram();
  CreateShader(program_yuv_, GL_VERTEX_SHADER, kVertexShader);
  CreateShader(program_yuv_, GL_FRAGMENT_SHADER, kFragmentShaderYUV);
  glLinkProgram(program_yuv_);
  AssertNoGLError();

  program_rgb_ = glCreateProgram();
  CreateShader(program_rgb_, GL_VERTEX_SHADER, kVertexShader);
  CreateShader(program_rgb_, GL_FRAGMENT_SHADER, kFragmentShaderRGB);
  glLinkProgram(program_rgb_);
  AssertNoGLError();

  // Assign vertex positions and texture coordinates to buffers for use in
  // shader program.
  static const float kVertices[] = {
    -1, 1, -1, -1, 0, 1, 0, -1,  // Position coordinates.
    0, 1, 0, -1, 1, 1, 1, -1,  // Position coordinates.
    0, 0, 0, 1, 1, 0, 1, 1,  // Texture coordinates.
    0, 0, 0, 1, 1, 0, 1, 1,  // Texture coordinates.
  };

  glGenBuffers(1, &buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, buffer_);
  glBufferData(GL_ARRAY_BUFFER, sizeof(kVertices), kVertices, GL_STATIC_DRAW);
  AssertNoGLError();
}

void MediaStreamVideoDemoInstance::CreateShader(
    GLuint program, GLenum type, const char* source) {
  GLuint shader = glCreateShader(type);
  GLint length = strlen(source) + 1;
  glShaderSource(shader, 1, &source, &length);
  glCompileShader(shader);
  glAttachShader(program, shader);
  glDeleteShader(shader);
}

void MediaStreamVideoDemoInstance::CreateTextures() {
  int32_t width = frame_size_.width();
  int32_t height = frame_size_.height();
  if (width == 0 || height == 0)
    return;
  if (texture_y_)
    glDeleteTextures(1, &texture_y_);
  if (texture_u_)
    glDeleteTextures(1, &texture_u_);
  if (texture_v_)
    glDeleteTextures(1, &texture_v_);
  if (texture_rgb_)
    glDeleteTextures(1, &texture_rgb_);
  texture_y_ = CreateTexture(width, height, 0, false);

  texture_u_ = CreateTexture(width / 2, height / 2, 1, false);
  texture_v_ = CreateTexture(width / 2, height / 2, 2, false);
  texture_rgb_ = CreateTexture(width, height, 3, true);
}

void MediaStreamVideoDemoInstance::ConfigureTrack() {
  const int32_t attrib_list[] = {
      PP_MEDIASTREAMVIDEOTRACK_ATTRIB_FORMAT, attrib_format_,
      PP_MEDIASTREAMVIDEOTRACK_ATTRIB_WIDTH, attrib_width_,
      PP_MEDIASTREAMVIDEOTRACK_ATTRIB_HEIGHT, attrib_height_,
      PP_MEDIASTREAMVIDEOTRACK_ATTRIB_NONE
    };
  video_track_.Configure(attrib_list, callback_factory_.NewCallback(
        &MediaStreamVideoDemoInstance::OnConfigure));
}

void MediaStreamVideoDemoInstance::OnConfigure(int32_t result) {
  video_track_.GetFrame(callback_factory_.NewCallbackWithOutput(
      &MediaStreamVideoDemoInstance::OnGetFrame));
}

void MediaStreamVideoDemoInstance::OnGetFrame(
    int32_t result, pp::VideoFrame frame) {
  //logmsg("Getting Frame");
  if (result != PP_OK)
    return;
  const char* data = static_cast<const char*>(frame.GetDataBuffer());
  pp::Size size;
  frame.GetSize(&size);

  if (size != frame_size_) {
    frame_size_ = size;
    CreateTextures();
  }

  is_bgra_ = (frame.GetFormat() == PP_VIDEOFRAME_FORMAT_BGRA);

  int32_t width = frame_size_.width();
  int32_t height = frame_size_.height();

  ///---->
  //&& imageCaptured==0
  if(is_bgra_==0 ) {

    //image = cv::Mat(height,width, CV_8UC4);
    image = cv::Mat(height+height/2,width, CV_8UC1);
    memcpy(image.ptr(), (void*) data, frame.GetDataBufferSize());
    // image.data = (unsigned char*)(data);
    // printf("0 Address of data is %p==%p\n", (void *)data,data);  
    // printf("0 Address of image data is %p==%p\n", (void *)image.data,image.data);  

    if( !image.empty() ) { 
      printf("size=%zu; width = %zu; height = %zu; is_bgra_=%zu",frame.GetDataBufferSize(),width,height,is_bgra_);

      printf(" image loaded\n");  
      imageCaptured = 1;
      DetectAndDisplay();
      //const char* d0 = static_cast<const char*>(frame.GetDataBuffer());

      //free((void*)data);
      //data = static_cast<const char*>(image.data);
    // printf("1 Address of data is %p==%p\n", (void *)data,data);  
    // printf("1 Address of image data is %p==%p\n", (void *)image.data,image.data);  

      //data =  reinterpret_cast<const char*>(image.data);
      data =  (char*)(image.data);

      printf("2 Address of data is %p==%p\n", (void *)data,data);  
      printf("2 Address of image data is %p==%p\n", (void *)image.data,image.data);  

    }  else { 
      printf(" --(!) No captured frame -- Break!\n"); 
    }

  }
  ///---->

  if (!is_bgra_) {
    glActiveTexture(GL_TEXTURE0);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                    GL_LUMINANCE, GL_UNSIGNED_BYTE, data);

    data += width * height;
    width /= 2;
    height /= 2;

    glActiveTexture(GL_TEXTURE1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                    GL_LUMINANCE, GL_UNSIGNED_BYTE, data);

    data += width * height;
    glActiveTexture(GL_TEXTURE2);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                    GL_LUMINANCE, GL_UNSIGNED_BYTE, data);
  } else {
    glActiveTexture(GL_TEXTURE3);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                    GL_BGRA_EXT, GL_UNSIGNED_BYTE, data);
  }

  if (is_painting_)
    needs_paint_ = true;
  else
    Render();

  video_track_.RecycleFrame(frame);
  if (need_config_) {
    ConfigureTrack();
    need_config_ = false;
  } else {
    video_track_.GetFrame(callback_factory_.NewCallbackWithOutput(
        &MediaStreamVideoDemoInstance::OnGetFrame));
  }

}

pp::Instance* MediaStreamVideoModule::CreateInstance(PP_Instance instance) {
  return new MediaStreamVideoDemoInstance(instance, this);
}

}  // anonymous namespace

namespace pp {
// Factory function for your specialization of the Module object.
Module* CreateModule() {
  return new MediaStreamVideoModule();
}
}  // namespace pp


//https://developer.chrome.com/native-client/cpp-api
//http://docs.opencv.org/modules/core/doc/basic_structures.html#mat
//http://docs.opencv.org/modules/core/doc/intro.html
//https://groups.google.com/forum/#!msg/native-client-discuss/fo3D7x62ocw/WemOy5zuJfMJ
