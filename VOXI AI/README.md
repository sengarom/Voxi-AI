# VOXI AI - Intelligent Multilingual Audio Processing

A modern, responsive web application for intelligent audio processing with speaker diarization, language detection, automatic speech recognition (ASR), and translation capabilities.

## 🌟 Features

### Core Audio Processing
- **Speaker Diarization**: Accurately identifies multiple speakers in an audio file
- **Language Detection**: Detects the spoken language of each segment automatically
- **Speech Recognition**: Converts speech to text efficiently using ASR technology
- **Translation**: Translates non-English audio into English seamlessly

### Technical Features
- **Audio Preprocessing**: Converts uploaded audio to WAV format, 16kHz, mono for consistent processing
- **Multi-Format Support**: Accepts .wav, .mp3, .flac, .ogg, .m4a audio files
- **Flexible Backend**: Ready to integrate with Flask backend powered by SpeechBrain and PyDub

### User Experience
- **Modern UI/UX**: Beautiful, responsive design with smooth animations
- **Drag & Drop Upload**: Intuitive file upload with drag and drop support
- **Real-time Processing**: Live progress tracking with status updates
- **Results Management**: Tabbed interface for viewing different processing results
- **Download Results**: Export processing results as text files

## 🚀 Getting Started

### Prerequisites
- A modern web browser (Chrome, Firefox, Safari, Edge)
- No server setup required - runs entirely in the browser

### Installation
1. Clone or download the project files
2. Open `index.html` in your web browser
3. Start using VOXI AI!

### File Structure
```
voxi-ai/
├── index.html          # Main HTML file
├── styles.css          # CSS styles and responsive design
├── script.js           # JavaScript functionality
└── README.md           # This file
```

## 📱 Usage Guide

### 1. Upload Audio File
- **Click Upload**: Click on the upload area to browse for audio files
- **Drag & Drop**: Drag audio files directly onto the upload area
- **Supported Formats**: .wav, .mp3, .flac, .ogg, .m4a
- **File Size Limit**: Maximum 100MB per file

### 2. Configure Processing Options
Select which features you want to apply to your audio:
- ✅ **Speaker Diarization** (enabled by default)
- ✅ **Language Detection** (enabled by default)
- ✅ **Speech Recognition** (enabled by default)
- ⬜ **Translation to English** (optional)

### 3. Process Audio
- Click "Process Audio" to start the analysis
- Watch real-time progress updates
- Processing typically takes 30-60 seconds depending on file size

### 4. View Results
Results are displayed in organized tabs:
- **Transcript**: Complete speech-to-text conversion with timestamps
- **Speakers**: Speaker identification and segmentation details
- **Languages**: Detected languages with confidence scores
- **Translation**: English translation (if enabled)

### 5. Download Results
- Click "Download Results" to save all processing data as a text file
- File includes transcript, speaker information, language detection, and translation

## 🎨 Design Features

### Modern UI Elements
- **Gradient Backgrounds**: Beautiful purple-blue gradients throughout
- **Smooth Animations**: CSS animations for enhanced user experience
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- **Interactive Elements**: Hover effects and smooth transitions

### Visual Components
- **Animated Audio Wave**: Visual representation of audio processing
- **Progress Indicators**: Real-time processing status with progress bars
- **Notification System**: Toast notifications for user feedback
- **Tabbed Interface**: Clean organization of different result types

## 🔧 Technical Implementation

### Frontend Technologies
- **HTML5**: Semantic markup and modern structure
- **CSS3**: Advanced styling with Flexbox and Grid layouts
- **JavaScript (ES6+)**: Modern JavaScript with async/await patterns
- **Font Awesome**: Icon library for visual elements
- **Google Fonts**: Inter font family for typography

### Key JavaScript Features
- **File Validation**: Type and size checking for uploaded files
- **Drag & Drop API**: Native HTML5 drag and drop functionality
- **Progress Simulation**: Realistic processing simulation with status updates
- **Results Generation**: Dynamic content generation based on processing options
- **Download Functionality**: Client-side file generation and download

### Responsive Design
- **Mobile-First**: Optimized for mobile devices
- **Breakpoints**: Responsive breakpoints at 768px and 480px
- **Touch-Friendly**: Large touch targets for mobile interaction
- **Flexible Layouts**: CSS Grid and Flexbox for adaptive layouts

## 🎯 Use Cases

### Business Applications
- **Meeting Transcription**: Convert meeting recordings to searchable text
- **Customer Service Analysis**: Analyze call center recordings
- **Podcast Processing**: Automatically transcribe and translate podcasts
- **Interview Analysis**: Process interview recordings with speaker identification

### Research & Development
- **Academic Research**: Process research interviews and focus groups
- **Language Studies**: Analyze multilingual conversations
- **Speech Analysis**: Study speech patterns and language usage
- **Accessibility**: Create transcripts for hearing-impaired users

### Content Creation
- **Video Subtitles**: Generate subtitles from video audio tracks
- **Documentation**: Create written records from audio content
- **Translation Services**: Translate audio content to different languages
- **Content Indexing**: Make audio content searchable and indexable

## 🔮 Future Enhancements

### Planned Features
- **Real Backend Integration**: Connect to actual SpeechBrain/PyDub backend
- **Batch Processing**: Process multiple files simultaneously
- **Advanced Analytics**: Detailed audio analysis and insights
- **API Integration**: RESTful API for programmatic access
- **Cloud Storage**: Integration with cloud storage providers

### Technical Improvements
- **WebAssembly**: Optimize processing with WebAssembly modules
- **Service Workers**: Offline functionality and caching
- **Real-time Streaming**: Process audio in real-time streams
- **Advanced Security**: Enhanced file validation and security measures

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📞 Support

For support or questions, please open an issue in the project repository.

---

**VOXI AI** - Making audio processing accessible to everyone through intelligent AI technology.
