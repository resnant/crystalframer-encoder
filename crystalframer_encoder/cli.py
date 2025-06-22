#!/usr/bin/env python3
"""Command-line interface for CrystalFramer Encoder"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path
from pymatgen.core import Structure
import crystalframer_encoder as cfe


def main():
    parser = argparse.ArgumentParser(
        description="Encode crystal structures using CrystalFramer models",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "structures",
        nargs="+",
        help="Path(s) to structure file(s) (CIF, POSCAR, etc.)"
    )
    
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to pretrained model checkpoint (.ckpt file)"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output file path (NPZ format). If not specified, prints to stdout"
    )
    
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)"
    )
    
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use (default: auto)"
    )
    
    parser.add_argument(
        "--embedding-dim",
        type=int,
        help="Override embedding dimension"
    )
    
    parser.add_argument(
        "--hparams",
        help="Path to hparams.yaml file (optional)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        # Load encoder
        if args.verbose:
            print(f"Loading model from {args.model}...", file=sys.stderr)
        
        encoder_kwargs = {}
        if args.embedding_dim:
            encoder_kwargs['embedding_dim'] = args.embedding_dim
        if args.hparams:
            encoder_kwargs['hparams_path'] = args.hparams
        if args.device:
            encoder_kwargs['device'] = args.device
            
        encoder = cfe.CrystalEncoder.from_pretrained(args.model, **encoder_kwargs)
        
        # Load structures
        structures = []
        structure_names = []
        
        for path in args.structures:
            path = Path(path)
            if not path.exists():
                print(f"Error: File not found: {path}", file=sys.stderr)
                sys.exit(1)
                
            try:
                structure = Structure.from_file(str(path))
                structures.append(structure)
                structure_names.append(path.stem)
                
                if args.verbose:
                    print(f"Loaded {path.name}: {structure.formula}", file=sys.stderr)
                    
            except Exception as e:
                print(f"Error loading {path}: {e}", file=sys.stderr)
                sys.exit(1)
        
        # Encode structures
        if args.verbose:
            print(f"Encoding {len(structures)} structures...", file=sys.stderr)
            
        embeddings = encoder.encode(
            structures, 
            batch_size=args.batch_size,
            show_progress=args.verbose
        )
        
        # Save or print results
        if args.output:
            # Save to NPZ file
            output_path = Path(args.output)
            np.savez(
                output_path,
                embeddings=embeddings.numpy(),
                names=structure_names,
                formulas=[s.formula for s in structures]
            )
            if args.verbose:
                print(f"Saved embeddings to {output_path}", file=sys.stderr)
        else:
            # Print to stdout as JSON
            results = {
                "embeddings": embeddings.numpy().tolist(),
                "names": structure_names,
                "formulas": [s.formula for s in structures],
                "shape": list(embeddings.shape)
            }
            print(json.dumps(results, indent=2))
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()