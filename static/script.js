// Global variables
let currentFile = null;
let processingResults = null;

// DOM elements
const uploadArea = document.getElementById('uploadArea');
const audioFileInput = document.getElementById('audioFile');
const fileInfo = document.getElementById('fileInfo');
const processingStatus = document.getElementById('processingStatus');
const results = document.getElementById('results');

// Navigation functionality
document.addEventListener('DOMContentLoaded', function() {
    // Mobile menu toggle
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    
    hamburger.addEventListener('click', function() {
        hamburger.classList.toggle('active');
        navMenu.classList.toggle('active');
    });

    // Close mobile menu when clicking on a link
    document.querySelectorAll('.nav-menu a').forEach(link => {
        link.addEventListener('click', () => {
            hamburger.classList.remove('active');
            navMenu.classList.remove('active');
        });
    });

    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Navbar background on scroll
    window.addEventListener('scroll', function() {
        const navbar = document.querySelector('.navbar');
        if (window.scrollY > 50) {
            navbar.style.background = 'rgba(0, 0, 0, 0.98)';
            navbar.style.boxShadow = '0 2px 20px rgba(0, 255, 65, 0.1)';
        } else {
            navbar.style.background = 'rgba(0, 0, 0, 0.95)';
            navbar.style.boxShadow = 'none';
        }
    });

    // Initialize upload functionality
    initializeUpload();
});

// Upload functionality
function initializeUpload() {
    // Click to upload
    uploadArea.addEventListener('click', () => {
        audioFileInput.click();
    });

    // File input change
    audioFileInput.addEventListener('change', handleFileSelect);

    // Drag and drop functionality
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type (robust: MIME or extension fallback)
    const allowedMimes = [
        'audio/wav', 'audio/x-wav',
        'audio/mpeg', /* common for .mp3 */
        'audio/mp3',
        'audio/flac',
        'audio/ogg',
        'audio/m4a', 'audio/x-m4a'
    ];
    const allowedExts = ['wav', 'mp3', 'flac', 'ogg', 'm4a'];
    const mimeOk = allowedMimes.includes((file.type || '').toLowerCase());
    const ext = (file.name.split('.').pop() || '').toLowerCase();
    const extOk = allowedExts.includes(ext);
    if (!mimeOk && !extOk) {
        showNotification('Please select a valid audio file (.wav, .mp3, .flac, .ogg, .m4a)', 'error');
        return;
    }

    // Validate file size (max 100MB)
    const maxSize = 100 * 1024 * 1024; // 100MB
    if (file.size > maxSize) {
        showNotification('File size must be less than 100MB', 'error');
        return;
    }

    currentFile = file;
    displayFileInfo(file);
    showNotification('File uploaded successfully!', 'success');
}

function displayFileInfo(file) {
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    
    uploadArea.style.display = 'none';
    fileInfo.style.display = 'block';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function removeFile() {
    currentFile = null;
    audioFileInput.value = '';
    uploadArea.style.display = 'block';
    fileInfo.style.display = 'none';
    processingStatus.style.display = 'none';
    results.style.display = 'none';
}

function resetUpload() {
    removeFile();
    scrollToSection('upload');
}

// Processing functionality
function processAudio() {
    if (!currentFile) {
        showNotification('Please select a file first', 'error');
        return;
    }

    // Get processing options
    const options = {
        speakerDiarization: document.getElementById('speakerDiarization').checked,
        languageDetection: document.getElementById('languageDetection').checked,
        speechRecognition: document.getElementById('speechRecognition').checked,
        translation: document.getElementById('translation').checked
    };

    // Show processing status
    fileInfo.style.display = 'none';
    processingStatus.style.display = 'block';
    
    // Use real backend processing
    uploadAndProcessAudio(options);
}

function simulateProcessing(options) {
    const statusMessages = [
        'Initializing processing pipeline...',
        'Converting audio to WAV format...',
        'Analyzing audio characteristics...',
        'Performing speaker diarization...',
        'Detecting languages...',
        'Converting speech to text...',
        'Generating transcript...',
        'Finalizing results...'
    ];

    const progressFill = document.getElementById('progressFill');
    const statusMessage = document.getElementById('statusMessage');
    
    let currentStep = 0;
    const totalSteps = statusMessages.length;
    
    const interval = setInterval(() => {
        if (currentStep < totalSteps) {
            const progress = ((currentStep + 1) / totalSteps) * 100;
            progressFill.style.width = progress + '%';
            statusMessage.textContent = statusMessages[currentStep];
            currentStep++;
        } else {
            clearInterval(interval);
            setTimeout(() => {
                showResults(options);
            }, 1000);
        }
    }, 800);
}

function showResults(options) {
    processingStatus.style.display = 'none';
    results.style.display = 'block';
    
    // Generate mock results based on options
    processingResults = generateMockResults(options);
    
    // Display results
    displayResults(processingResults);
    
    // Scroll to results
    results.scrollIntoView({ behavior: 'smooth' });
}

function generateMockResults(options) {
    const results = {
        transcript: '',
        speakers: [],
        languages: [],
        translation: ''
    };

    if (options.speechRecognition) {
        results.transcript = `[00:00:00] Speaker 1: Hello, welcome to our meeting today. I'm excited to discuss the new project with all of you.

[00:00:05] Speaker 2: Thank you for organizing this. I've been looking forward to this discussion.

[00:00:10] Speaker 1: Great! Let's start with the overview. As you all know, we're working on the VOXI AI platform.

[00:00:15] Speaker 3: Yes, I've been reviewing the documentation. The audio processing capabilities are quite impressive.

[00:00:20] Speaker 2: Absolutely! The speaker diarization feature is particularly useful for our use case.

[00:00:25] Speaker 1: That's exactly right. We've integrated SpeechBrain and PyDub for robust processing.

[00:00:30] Speaker 3: How does the language detection work with multiple speakers?

[00:00:35] Speaker 1: Excellent question. The system can detect different languages within the same conversation.`;
    }

    if (options.speakerDiarization) {
        results.speakers = [
            { id: 1, name: 'Speaker 1', duration: '00:02:35', segments: 8 },
            { id: 2, name: 'Speaker 2', duration: '00:01:20', segments: 3 },
            { id: 3, name: 'Speaker 3', duration: '00:01:15', segments: 2 }
        ];
    }

    if (options.languageDetection) {
        results.languages = [
            { language: 'English', confidence: 0.98, duration: '00:02:35' },
            { language: 'Spanish', confidence: 0.85, duration: '00:00:45' },
            { language: 'French', confidence: 0.92, duration: '00:00:30' }
        ];
    }

    if (options.translation) {
        results.translation = `[00:00:00] Speaker 1: Hello, welcome to our meeting today. I'm excited to discuss the new project with all of you.

[00:00:05] Speaker 2: Thank you for organizing this. I've been looking forward to this discussion.

[00:00:10] Speaker 1: Great! Let's start with the overview. As you all know, we're working on the VOXI AI platform.

[00:00:15] Speaker 3: Yes, I've been reviewing the documentation. The audio processing capabilities are quite impressive.

[00:00:20] Speaker 2: Absolutely! The speaker diarization feature is particularly useful for our use case.

[00:00:25] Speaker 1: That's exactly right. We've integrated SpeechBrain and PyDub for robust processing.

[00:00:30] Speaker 3: How does the language detection work with multiple speakers?

[00:00:35] Speaker 1: Excellent question. The system can detect different languages within the same conversation.`;
    }

    return results;
}

function displayResults(results) {
    // Display transcript
    const transcriptContent = document.getElementById('transcriptContent');
    transcriptContent.innerHTML = `<pre>${results.transcript || 'No transcript available'}</pre>`;

    // Display speakers
    const speakersContent = document.getElementById('speakersContent');
    if (results.speakers.length > 0) {
        speakersContent.innerHTML = results.speakers.map(speaker => `
            <div class="speaker-item">
                <h4>${speaker.name}</h4>
                <p>Duration: ${speaker.duration}</p>
                <p>Segments: ${speaker.segments}</p>
            </div>
        `).join('');
    } else {
        speakersContent.innerHTML = '<p>No speaker information available</p>';
    }

    // Display languages
    const languagesContent = document.getElementById('languagesContent');
    if (results.languages.length > 0) {
        languagesContent.innerHTML = results.languages.map(lang => `
            <div class="language-item">
                <h4>${lang.language}</h4>
                <p>Confidence: ${(lang.confidence * 100).toFixed(1)}%</p>
                <p>Duration: ${lang.duration}</p>
            </div>
        `).join('');
    } else {
        languagesContent.innerHTML = '<p>No language information available</p>';
    }

    // Display translation
    const translationContent = document.getElementById('translationContent');
    translationContent.innerHTML = `<pre>${results.translation || 'No translation available'}</pre>`;
}

// Tab functionality
function showTab(tabName, clickEvent) {
    // Hide all tab panes
    const tabPanes = document.querySelectorAll('.tab-pane');
    tabPanes.forEach(pane => pane.classList.remove('active'));

    // Remove active class from all tab buttons
    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(btn => btn.classList.remove('active'));

    // Show selected tab pane
    const selectedPane = document.getElementById(tabName);
    if (selectedPane) {
        selectedPane.classList.add('active');
    }

    // Add active class to clicked button or find the button if called programmatically
    let targetButton;
    if (clickEvent && clickEvent.target) {
        targetButton = clickEvent.target;
    } else {
        // If called programmatically, find the corresponding button
        targetButton = Array.from(tabButtons).find(
            btn => btn.getAttribute('onclick').includes(`'${tabName}'`)
        );
    }
    if (targetButton) {
        targetButton.classList.add('active');
    }
}

// Utility functions
function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <span>${message}</span>
        </div>
    `;

    // Add styles
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#6366f1'};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        z-index: 10000;
        transform: translateX(100%);
        transition: transform 0.3s ease;
        max-width: 300px;
    `;

    // Add to page
    document.body.appendChild(notification);

    // Animate in
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 100);

    // Remove after 5 seconds
    setTimeout(() => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 5000);
}

function downloadResults() {
    if (!processingResults) {
        showNotification('No results to download', 'error');
        return;
    }

    // Create downloadable content
    let content = 'VOXI AI Processing Results\n';
    content += '==========================\n\n';
    
    if (processingResults.transcript) {
        content += 'TRANSCRIPT:\n';
        content += processingResults.transcript + '\n\n';
    }
    
    if (processingResults.speakers.length > 0) {
        content += 'SPEAKERS:\n';
        processingResults.speakers.forEach(speaker => {
            content += `${speaker.name}: ${speaker.duration} (${speaker.segments} segments)\n`;
        });
        content += '\n';
    }
    
    if (processingResults.languages && processingResults.languages.length > 0) {
        content += 'LANGUAGES:\n';
        processingResults.languages.forEach(lang => {
            content += `${lang.language}: ${(lang.confidence * 100).toFixed(1)}% confidence, ${lang.duration}\n`;
        });
        content += '\n';
    }
    // Fallback for overall language field from backend
    if ((!processingResults.languages || processingResults.languages.length === 0) && processingResults.language) {
        content += 'LANGUAGE (overall):\n';
        content += String(processingResults.language).toUpperCase() + '\n\n';
    }
    
    // Include per-segment translations when available
    const segsWithTranslations = Array.isArray(processingResults.speakers) ? processingResults.speakers.filter(s => (s.translation || '').trim()) : [];
    if (segsWithTranslations.length > 0) {
        content += 'TRANSLATION (per speaker):\n';
        segsWithTranslations.forEach(s => {
            const spk = (s.speaker || 'Speaker');
            content += `[${spk}] ${s.translation}\n`;
        });
        content += '\n';
    }
    if (processingResults.translation) {
        content += 'TRANSLATION (combined):\n';
        content += processingResults.translation + '\n';
    }

    // Create and download file
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `voxi-ai-results-${new Date().toISOString().slice(0, 10)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showNotification('Results downloaded successfully!', 'success');
}

// --- Real audio upload and backend integration ---
function uploadAndProcessAudio(options) {
    if (!currentFile) {
        showNotification('Please select a file first', 'error');
        return;
    }
    fileInfo.style.display = 'none';
    processingStatus.style.display = 'block';
    results.style.display = 'none';
    const progressFill = document.getElementById('progressFill');
    const statusMessage = document.getElementById('statusMessage');
    progressFill.style.width = '10%';
    statusMessage.textContent = 'Uploading audio...';

    const formData = new FormData();
    formData.append('file', currentFile);
    // Pass options to backend (backend may ignore them)
    try {
        if (options && typeof options === 'object') {
            // No additional options to add
            formData.append('options', JSON.stringify(options));
        }
    } catch (_) {}

    fetch('/process_audio', {
        method: 'POST',
        body: formData
    })
    .then(async response => {
        progressFill.style.width = '60%';
        statusMessage.textContent = 'Processing audio...';
        if (!response.ok) {
            let errText = await response.text();
            throw new Error(`Server error: ${response.status} ${errText}`);
        }
        return response.json();
    })
    .then(data => {
        progressFill.style.width = '100%';
        statusMessage.textContent = 'Done!';
        setTimeout(() => {
            processingStatus.style.display = 'none';
            // Respect translation option on UI
            try {
                const translateChecked = document.getElementById('translation').checked;
                if (!translateChecked) {
                    data.translation = '';
                    if (Array.isArray(data.speakers)) {
                        data.speakers = data.speakers.map(s => ({
                            ...s,
                            translation: ''
                        }));
                    }
                }
            } catch (_) {}
            displayBackendResults(data);
        }, 500);
        console.log('Backend response:', data);
    })
    .catch(error => {
        processingStatus.style.display = 'none';
        showNotification('Error: ' + error.message, 'error');
        console.error('Upload/process error:', error);
    });
}

function displayBackendResults(data) {
    // Store for download
    processingResults = data;

    // Show results container (keeps tabs)
    results.style.display = 'block';

    // Fill transcript (optionally show English per speaker)
    const transcriptContent = document.getElementById('transcriptContent');
    const speakersArr = Array.isArray(data.speakers) ? data.speakers : [];
    const showEnglish = (() => { try { return document.getElementById('showEnglishTranscript').checked; } catch(_) { return false; } })();
    if (speakersArr.length) {
        const blocks = speakersArr.map(s => {
            const spk = (s.speaker || 'Speaker').toString();
            const start = (s.start ?? '').toString();
            const end = (s.end ?? '').toString();
            const baseText = (s.transcript || '').toString().trim();
            const engText = (s.translation || '').toString().trim();
            const text = showEnglish && engText ? engText : baseText;
            return `<div class="speaker-item"><h4>${spk} <small style="color:#94a3b8">(${start}s–${end}s)</small></h4><p>${text || '<em>No transcript</em>'}</p></div>`;
        }).join('');
        transcriptContent.innerHTML = blocks;
    } else {
        transcriptContent.innerHTML = `<pre>${(data.transcript || '').trim() || 'No transcript available'}</pre>`;
    }

    // Fill speakers
    const speakersContent = document.getElementById('speakersContent');
    const speakers = Array.isArray(data.speakers) ? data.speakers : [];
    if (speakers.length) {
        speakersContent.innerHTML = speakers.map(s => {
            const start = (s.start ?? s.start_time ?? '').toString();
            const end = (s.end ?? s.end_time ?? '').toString();
            const text = (s.transcript || s.transcription || '').toString();
            const speaker = (s.speaker || 'Speaker').toString();
            return `
            <div class="speaker-item">
                <h4>${speaker}</h4>
                <p>Start: ${start}s | End: ${end}s</p>
                <p>${text ? text : '<em>No segment transcript</em>'}</p>
            </div>`;
        }).join('');
    } else {
        speakersContent.innerHTML = '<p>No speaker information available</p>';
    }

    // Fill languages (overall language only; per-segment not provided by API)
    const languagesContent = document.getElementById('languagesContent');
    const lang = (data.language || 'unknown').toString();
    languagesContent.innerHTML = `
        <div class="language-item">
            <h4>${lang.toUpperCase()}</h4>
            <p>Detected overall language code</p>
        </div>
    `;

    // --- Translation Tab Logic ---
    const translationContent = document.getElementById('translationContent');
    const translationTabButton = Array.from(document.querySelectorAll('.results-tabs .tab-btn')).find(btn => btn.getAttribute('onclick').includes("'translation'"));
    
    const perSegmentTranslations = (Array.isArray(data.speakers) ? data.speakers : []).filter(s => (s.translation || '').trim());
    const fullTranslation = (data.translation || '').trim();
    
    // Determine if there is any meaningful translation to show
    const hasTranslation = perSegmentTranslations.length > 0 || (fullTranslation && fullTranslation !== data.transcript);

    if (hasTranslation) {
        // 1. Populate the translation tab content
        if (perSegmentTranslations.length > 0) {
            translationContent.innerHTML = perSegmentTranslations.map(s => {
                const spk = (s.speaker || 'Speaker').toString();
                const text = (s.translation || '').toString();
                return `<div class="speaker-item"><h4>${spk}</h4><p>${text}</p></div>`;
            }).join('');
        } else {
            translationContent.innerHTML = `<pre>${fullTranslation}</pre>`;
        }

        // 2. Enable the translation tab button
        if (translationTabButton) {
            translationTabButton.removeAttribute('disabled');
            translationTabButton.classList.remove('disabled');
        }

        // 3. If user requested translation, switch to that tab
        if (document.getElementById('translation').checked) {
            showTab('translation');
        }
    } else {
        // No translation available
        translationContent.innerHTML = '<p>No translation available or text was already in English.</p>';
        
        // Disable the tab
        if (translationTabButton) {
            translationTabButton.setAttribute('disabled', 'true');
            translationTabButton.classList.add('disabled');
        }
    }

    // Scroll into view
    results.scrollIntoView({ behavior: 'smooth' });
}

// Process Audio button triggers real upload in processAudio()

// Add some CSS for notifications
const notificationStyles = document.createElement('style');
notificationStyles.textContent = `
    .notification-content {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .speaker-item, .language-item {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .speaker-item h4, .language-item h4 {
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .speaker-item p, .language-item p {
        color: #64748b;
        margin: 0.25rem 0;
    }
    
    pre {
        white-space: pre-wrap;
        word-wrap: break-word;
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
        color: #1e293b;
    }
`;
document.head.appendChild(notificationStyles);