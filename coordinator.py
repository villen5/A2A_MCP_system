#!/usr/bin/env python
"""
Simple A2A Coordinator for Two OpenAI Agents
"""

import argparse
import time
from python_a2a import A2AClient

def main():
    parser = argparse.ArgumentParser(description="Simple A2A Agent Coordinator")
    parser.add_argument("--prompt", type=str, default="What are the prices of the 3 most expensive Nike shoes?",
                      help="Initial prompt to start the conversation")
    parser.add_argument("--turns", type=int, default=3,
                      help="Number of back-and-forth turns (default: 2)")
    args = parser.parse_args()
    
    # Connect to both agents
    agent1 = A2AClient("http://localhost:5000")
    agent2 = A2AClient("http://localhost:5002")
    
    # Start conversation
    current_message = args.prompt
    print(f"\n[HUMAN] Initial prompt: {current_message}\n")
    
    for i in range(args.turns):
        # Agent 1's turn
        print(f"[TURN {i+1}] Sending to Agent 1: {current_message[:50]}...")
        response1 = agent1.ask(current_message)
        print(f"\n[AGENT 1] {response1}\n")
        
        # Agent 2's turn
        print(f"[TURN {i+1}] Sending to Agent 2: {response1[:50]}...")
        response2 = agent2.ask(response1)
        print(f"\n[AGENT 2] {response2}\n")
        
        # Update for next turn
        current_message = response2
        
        # Short pause between turns
        time.sleep(1)
    
    print("Conversation completed!")

if __name__ == "__main__":
    main()
