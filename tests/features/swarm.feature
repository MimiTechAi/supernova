Feature: Liquid Swarm Big Bang Cycle
  As a Liquid Swarm operator
  I want to execute a massively parallel analysis
  So that I get results in minutes instead of weeks

  Scenario: Full graph cycle 1 → 50 → 1
    Given a swarm with 50 analysis tasks
    And all APIs are mocked with zero cost
    When the Big Bang is ignited
    Then exactly 50 parallel workers are spawned
    And all 50 workers return a valid TaskResult
    And the reduce phase aggregates to a single report
    And the total cost is 0.10 USD

  Scenario: A single rogue worker does not crash the swarm
    Given a swarm with 10 analysis tasks
    And worker 5 returns impossible data with market_share 150 percent
    When the Big Bang is ignited
    Then the final report contains exactly 9 valid results
    And 1 result is flagged as invalid

  Scenario: Timeout of a worker isolates the damage
    Given a swarm with 10 analysis tasks
    And worker 3 hangs for 30 seconds
    When the Big Bang is ignited
    Then the final report contains 10 results
    And worker 3 has status timeout
    And the other 9 workers have status success
