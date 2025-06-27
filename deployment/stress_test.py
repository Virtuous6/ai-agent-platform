#!/usr/bin/env python3
"""
Cloud Run Stress Test Script
Tests auto-scaling capabilities of the AI agent platform
"""

import asyncio
import aiohttp
import time
import json
from concurrent.futures import ThreadPoolExecutor

class CloudRunStressTester:
    def __init__(self, service_url: str):
        self.service_url = service_url.rstrip('/')
        self.results = []
    
    async def send_request(self, session, request_id):
        """Send a single test request."""
        start_time = time.time()
        try:
            async with session.post(
                f"{self.service_url}/slack/events",
                json={
                    "type": "event_callback",
                    "event": {
                        "type": "app_mention",
                        "text": f"<@bot> stress test request {request_id}",
                        "user": "stress_test_user",
                        "channel": "stress_test"
                    }
                },
                timeout=30
            ) as response:
                end_time = time.time()
                
                result = {
                    "request_id": request_id,
                    "status_code": response.status,
                    "response_time": end_time - start_time,
                    "success": response.status == 200
                }
                
                self.results.append(result)
                return result
                
        except Exception as e:
            end_time = time.time()
            result = {
                "request_id": request_id,
                "status_code": 0,
                "response_time": end_time - start_time,
                "success": False,
                "error": str(e)
            }
            self.results.append(result)
            return result
    
    async def run_stress_test(self, total_requests=1000, concurrent_requests=50):
        """Run the stress test."""
        print(f"🚀 Starting stress test against {self.service_url}")
        print(f"📊 Total requests: {total_requests}")
        print(f"⚡ Concurrent: {concurrent_requests}")
        
        async with aiohttp.ClientSession() as session:
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(concurrent_requests)
            
            async def limited_request(request_id):
                async with semaphore:
                    return await self.send_request(session, request_id)
            
            # Send all requests
            tasks = [limited_request(i) for i in range(total_requests)]
            await asyncio.gather(*tasks)
        
        self.analyze_results()
    
    def analyze_results(self):
        """Analyze stress test results."""
        if not self.results:
            print("❌ No results to analyze")
            return
        
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]
        
        response_times = [r["response_time"] for r in successful]
        
        print("\n📈 STRESS TEST RESULTS:")
        print(f"Total Requests: {len(self.results)}")
        print(f"Successful: {len(successful)} ({len(successful)/len(self.results)*100:.1f}%)")
        print(f"Failed: {len(failed)} ({len(failed)/len(self.results)*100:.1f}%)")
        
        if response_times:
            print(f"\n⏱️  RESPONSE TIMES:")
            print(f"Average: {sum(response_times)/len(response_times):.2f}s")
            print(f"Min: {min(response_times):.2f}s")
            print(f"Max: {max(response_times):.2f}s")
        
        # Save detailed results
        with open("stress_test_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        print("\n💾 Detailed results saved to stress_test_results.json")

async def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python stress_test.py <cloud-run-service-url>")
        sys.exit(1)
    
    service_url = sys.argv[1]
    tester = CloudRunStressTester(service_url)
    
    # Run stress test
    await tester.run_stress_test(
        total_requests=500,
        concurrent_requests=25
    )

if __name__ == "__main__":
    asyncio.run(main())