"""
HybridMusicDataset - Data loading and preprocessing for Audio + Lyrics
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import librosa
import warnings
warnings.filterwarnings('ignore')


class HybridMusicDataset(Dataset):
    """
    PyTorch Dataset for Hybrid Audio-Lyrics Music Data
    Supports loading from Q1-Q4 folders and CSV metadata
    """
    
    def __init__(self, 
                 audio_root='data/audio',
                 lyrics_root='data/lyrics',
                 split='train',
                 tvt_folder='tvt_70_15_15',
                 n_mfcc=20,
                 max_length=500,
                 tfidf_max_features=300,
                 device='cpu'):
        """
        Args:
            audio_root: Root directory for audio data
            lyrics_root: Root directory for lyrics data
            split: 'train', 'val', or 'test'
            tvt_folder: Train/Val/Test split folder
            n_mfcc: Number of MFCC features
            max_length: Maximum length for audio features
            tfidf_max_features: Maximum TF-IDF features
            device: Device to place tensors on
        """
        self.audio_root = audio_root
        self.lyrics_root = lyrics_root
        self.split = split
        self.n_mfcc = n_mfcc
        self.max_length = max_length
        self.tfidf_max_features = tfidf_max_features
        self.device = device
        
        # Load metadata and values from CSVs
        self.audio_metadata, self.audio_values = self._load_audio_csvs(audio_root)
        self.lyrics_metadata, self.lyrics_values = self._load_lyrics_csvs(lyrics_root)
        
        # Create unified dataset
        self.data = self._merge_data()
        
        # Apply train/val/test split
        self.data = self._apply_split(split, tvt_folder)
        
        # Initialize TF-IDF vectorizer for lyrics
        self.tfidf_vectorizer = None
        self._initialize_tfidf()
        
        # Print dataset info
        self._print_dataset_info()
        
    def _load_audio_csvs(self, audio_root):
        """Load audio metadata and values CSVs"""
        metadata_path = os.path.join(audio_root, 'merge_audio_complete_metadata.csv')
        values_path = os.path.join(audio_root, 'merge_audio_complete_av_values.csv')
        
        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
        else:
            metadata = pd.DataFrame()
            
        if os.path.exists(values_path):
            values = pd.read_csv(values_path)
        else:
            values = pd.DataFrame()
            
        return metadata, values
    
    def _load_lyrics_csvs(self, lyrics_root):
        """Load lyrics metadata and values CSVs"""
        metadata_path = os.path.join(lyrics_root, 'merge_lyrics_complete_metadata.csv')
        values_path = os.path.join(lyrics_root, 'merge_lyrics_complete_av_values.csv')
        
        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
        else:
            metadata = pd.DataFrame()
            
        if os.path.exists(values_path):
            values = pd.read_csv(values_path)
        else:
            values = pd.DataFrame()
            
        return metadata, values
    
    def _merge_data(self):
        """Merge audio and lyrics data, scan Q1-Q4 folders"""
        data_list = []
        
        # Scan all quadrant folders for files
        for quadrant in ['Q1', 'Q2', 'Q3', 'Q4']:
            # Audio files
            audio_dir = os.path.join(self.audio_root, quadrant)
            lyrics_dir = os.path.join(self.lyrics_root, quadrant)
            
            # Get all files in quadrant
            if os.path.exists(lyrics_dir):
                lyrics_files = [f for f in os.listdir(lyrics_dir) if f.endswith('.txt')]
                
                for lyrics_file in lyrics_files:
                    # Extract ID from filename
                    file_id = lyrics_file.replace('.txt', '')
                    
                    # Look for corresponding audio (if exists)
                    audio_file = None
                    if os.path.exists(audio_dir):
                        # Try to find matching audio file (could be .wav, .mp3, etc.)
                        for ext in ['.wav', '.mp3', '.flac', '.ogg']:
                            potential_audio = os.path.join(audio_dir, file_id + ext)
                            if os.path.exists(potential_audio):
                                audio_file = potential_audio
                                break
                    
                    data_list.append({
                        'id': file_id,
                        'lyrics_path': os.path.join(lyrics_dir, lyrics_file),
                        'audio_path': audio_file,
                        'label': quadrant,
                        'label_idx': int(quadrant[1]) - 1  # Q1->0, Q2->1, Q3->2, Q4->3
                    })
        
        return pd.DataFrame(data_list)
    
    def _apply_split(self, split, tvt_folder):
        """Apply train/val/test split (70/20/10)"""
        tvt_path = os.path.join(self.lyrics_root, 'tvt_dataframes', tvt_folder)
        
        # Try to load pre-existing split
        split_file = os.path.join(tvt_path, f'{split}.csv')
        if os.path.exists(split_file):
            split_df = pd.read_csv(split_file)
            # Filter current data by IDs in split
            if 'id' in split_df.columns:
                return self.data[self.data['id'].isin(split_df['id'])]
        
        # Otherwise, create split dynamically
        if split == 'train':
            return self.data.sample(frac=0.7, random_state=42)
        elif split == 'val':
            remaining = self.data.drop(self.data.sample(frac=0.7, random_state=42).index)
            return remaining.sample(frac=0.667, random_state=42)  # 20% of total
        else:  # test
            remaining = self.data.drop(self.data.sample(frac=0.7, random_state=42).index)
            return remaining.drop(remaining.sample(frac=0.667, random_state=42).index)
    
    def _initialize_tfidf(self):
        """Initialize TF-IDF vectorizer on all lyrics"""
        all_lyrics = []
        for _, row in self.data.iterrows():
            lyrics_text = self._read_lyrics(row['lyrics_path'])
            all_lyrics.append(lyrics_text)
        
        if len(all_lyrics) > 0:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.tfidf_max_features,
                stop_words='english',
                lowercase=True
            )
            self.tfidf_vectorizer.fit(all_lyrics)
    
    def _read_lyrics(self, lyrics_path):
        """Read lyrics from text file"""
        try:
            with open(lyrics_path, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return ""
    
    def _load_audio_features(self, audio_path):
        """Load audio and extract MFCC features"""
        if audio_path is None or not os.path.exists(audio_path):
            # Return zero features if audio doesn't exist
            return np.zeros((self.n_mfcc * self.max_length,))
        
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=22050, duration=30)
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            
            # Flatten and pad/truncate to fixed length
            mfccs_flat = mfccs.flatten()
            if len(mfccs_flat) < self.n_mfcc * self.max_length:
                mfccs_flat = np.pad(mfccs_flat, (0, self.n_mfcc * self.max_length - len(mfccs_flat)))
            else:
                mfccs_flat = mfccs_flat[:self.n_mfcc * self.max_length]
            
            return mfccs_flat
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return np.zeros((self.n_mfcc * self.max_length,))
    
    def _load_lyrics_features(self, lyrics_path):
        """Load lyrics and convert to TF-IDF features"""
        lyrics_text = self._read_lyrics(lyrics_path)
        
        if self.tfidf_vectorizer is not None and lyrics_text:
            tfidf_features = self.tfidf_vectorizer.transform([lyrics_text]).toarray()[0]
            return tfidf_features
        else:
            return np.zeros((self.tfidf_max_features,))
    
    def _print_dataset_info(self):
        """Print dataset summary statistics"""
        print("\n" + "="*60)
        print(f"DATASET INFO - {self.split.upper()} SPLIT")
        print("="*60)
        print(f"Total Samples: {len(self.data)}")
        print("\nClass Distribution:")
        class_counts = self.data['label'].value_counts().sort_index()
        for label, count in class_counts.items():
            print(f"  {label}: {count} samples ({count/len(self.data)*100:.1f}%)")
        
        print("\nMissing Value Statistics:")
        print(f"  Missing Audio Files: {self.data['audio_path'].isna().sum()}")
        print(f"  Missing Lyrics Files: {self.data['lyrics_path'].isna().sum()}")
        print("="*60 + "\n")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        row = self.data.iloc[idx]
        
        # Load audio features
        audio_features = self._load_audio_features(row['audio_path'])
        
        # Load lyrics features
        lyrics_features = self._load_lyrics_features(row['lyrics_path'])
        
        # Convert to tensors
        audio_tensor = torch.FloatTensor(audio_features)
        lyrics_tensor = torch.FloatTensor(lyrics_features)
        label = torch.LongTensor([row['label_idx']])
        
        return {
            'audio': audio_tensor,
            'lyrics': lyrics_tensor,
            'label': label.squeeze()
        }


def get_dataloaders(audio_root='data/audio',
                    lyrics_root='data/lyrics',
                    batch_size=32,
                    tvt_folder='tvt_70_15_15',
                    n_mfcc=20,
                    max_length=500,
                    tfidf_max_features=300,
                    device='cpu',
                    num_workers=0):
    """
    Create train/val/test dataloaders with 70/20/10 split
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print("\n🔄 Creating Dataloaders...")
    
    # Create datasets for each split
    train_dataset = HybridMusicDataset(
        audio_root=audio_root,
        lyrics_root=lyrics_root,
        split='train',
        tvt_folder=tvt_folder,
        n_mfcc=n_mfcc,
        max_length=max_length,
        tfidf_max_features=tfidf_max_features,
        device=device
    )
    
    val_dataset = HybridMusicDataset(
        audio_root=audio_root,
        lyrics_root=lyrics_root,
        split='val',
        tvt_folder=tvt_folder,
        n_mfcc=n_mfcc,
        max_length=max_length,
        tfidf_max_features=tfidf_max_features,
        device=device
    )
    
    test_dataset = HybridMusicDataset(
        audio_root=audio_root,
        lyrics_root=lyrics_root,
        split='test',
        tvt_folder=tvt_folder,
        n_mfcc=n_mfcc,
        max_length=max_length,
        tfidf_max_features=tfidf_max_features,
        device=device
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    print(f"✅ Dataloaders created successfully!")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}\n")
    
    return train_loader, val_loader, test_loader
