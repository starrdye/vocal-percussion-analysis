import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def create_poster():
    fig, axs = plt.subplots(2, 2, figsize=(20, 14))
    
    # Increase the space between the title and the plots, and between the plots themselves
    fig.subplots_adjust(top=0.9, hspace=0.25, wspace=0.15)
    fig.suptitle('Phase 2: Supervised Beatbox Classification', fontsize=26, fontweight='bold', y=0.96)
    
    img1 = mpimg.imread('public/phase2_confusion_cnn.png')
    img2 = mpimg.imread('public/phase2_per_sound_comparison.png')
    img3 = mpimg.imread('public/phase2_feature_importance.png')
    img4 = mpimg.imread('public/phase2_lopo_per_participant.png')

    # Remove all margins from subplots
    axs[0, 0].imshow(img1)
    axs[0, 0].axis('off')
    axs[0, 0].set_title('CNN Confusion Matrix', fontsize=20, pad=20)

    axs[0, 1].imshow(img2)
    axs[0, 1].axis('off')
    axs[0, 1].set_title('Per-Sound Accuracy Comparison', fontsize=20, pad=20)

    axs[1, 0].imshow(img3)
    axs[1, 0].axis('off')
    axs[1, 0].set_title('Random Forest Feature Importance', fontsize=20, pad=20)

    axs[1, 1].imshow(img4)
    axs[1, 1].axis('off')
    axs[1, 1].set_title('LOPO Per-Participant Accuracy', fontsize=20, pad=20)

    # Save to public directory
    plt.savefig('public/poster.png', dpi=150, bbox_inches='tight', pad_inches=0.4, facecolor='white')
    print("Poster created successfully at public/poster.png")

if __name__ == "__main__":
    create_poster()
