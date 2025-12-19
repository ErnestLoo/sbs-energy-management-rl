#ifndef SMALL_CELL_ENERGY_MODEL_H
#define SMALL_CELL_ENERGY_MODEL_H

#include "ns3/object.h"
#include "ns3/ptr.h"
#include "ns3/type-id.h"
#include "ns3/simulator.h"
#include "ns3/device-energy-model.h"
#include "ns3/energy-source.h"
#include <fstream>
#include <vector>
#include <string>

namespace ns3 {

class SmallCellEnergyModel : public energy::DeviceEnergyModel
{
public:
    enum SmallCellState {
        ACTIVE = 0,
        SM1 = 1,
        SM2 = 2,
        SM3 = 3
    };

    static TypeId GetTypeId(void)
    {
        static TypeId tid = TypeId("ns3::SmallCellEnergyModel")
            .SetParent<energy::DeviceEnergyModel>()
            .SetGroupName("Energy")
            .AddConstructor<SmallCellEnergyModel>();
        return tid;
    }

    SmallCellEnergyModel()
        : m_state(ACTIVE),
          m_totalEnergy(0.0),
          m_lastUpdateTime(Seconds(0)),
          m_isTransitioning(false),
          m_transitionEndTime(Seconds(0)),
          m_nodeId(0)
    {
        m_powerConsumption[ACTIVE] = 20.7;
        m_powerConsumption[SM1] = 15.0;
        m_powerConsumption[SM2] = 10.0;
        m_powerConsumption[SM3] = 3.36;

        m_txPower[ACTIVE] = 30.0;
        m_txPower[SM1] = 20.0;
        m_txPower[SM2] = 10.0;
        m_txPower[SM3] = 0.0;

        m_transitionTime[ACTIVE] = 0.0;
        m_transitionTime[SM1] = 0.01;
        m_transitionTime[SM2] = 0.05;
        m_transitionTime[SM3] = 0.1;
    }

    virtual ~SmallCellEnergyModel() {}

    // Required by DeviceEnergyModel
    virtual void SetEnergySource(Ptr<energy::EnergySource> source) override
    {
        m_energySource = source;
    }

    virtual void ChangeState(int newState) override
    {
        SetState(static_cast<SmallCellState>(newState));
    }

    virtual void HandleEnergyDepletion() override
    {
        // Handle energy depletion if needed
    }

    virtual void HandleEnergyRecharged() override
    {
        // Handle energy recharged if needed
    }

    virtual void HandleEnergyChanged() override
    {
        // Handle energy changed if needed
    }

    virtual double GetTotalEnergyConsumption() const override
    {
        return m_totalEnergy;
    }

    void SetNodeId(uint32_t nodeId)
    {
        m_nodeId = nodeId;
    }

    uint32_t GetNodeId(void) const
    {
        return m_nodeId;
    }

    void SetState(SmallCellState newState)
    {
        Time now = Simulator::Now();
        UpdateEnergyConsumption();

        if (newState != m_state)
        {
            m_isTransitioning = true;
            m_transitionEndTime = now + Seconds(m_transitionTime[newState]);
            m_stateLog.push_back({now.GetSeconds(), m_state, newState});
        }

        m_state = newState;
    }

    SmallCellState GetState(void) const
    {
        return m_state;
    }

    double GetTotalPowerConsumption(void) const
    {
        return m_powerConsumption[m_state];
    }

    double GetTransmissionPower(void) const
    {
        return m_txPower[m_state];
    }

    bool IsTransitioning(void) const
    {
        if (m_isTransitioning)
        {
            Time now = Simulator::Now();
            if (now >= m_transitionEndTime)
            {
                const_cast<SmallCellEnergyModel*>(this)->m_isTransitioning = false;
            }
        }
        return m_isTransitioning;
    }

    void FlushFinalState(void)
    {
        UpdateEnergyConsumption();
    }

    void ExportStateTimeToCsv(int sbsId)
    {
        std::string filename = "sbs_" + std::to_string(sbsId) + "_state_log.csv";
        std::ofstream file(filename);
        
        if (file.is_open())
        {
            file << "time,from_state,to_state\n";
            for (const auto& entry : m_stateLog)
            {
                file << entry.time << "," << entry.fromState << "," << entry.toState << "\n";
            }
            file.close();
        }
    }

private:
    void UpdateEnergyConsumption(void)
    {
        Time now = Simulator::Now();
        double deltaTime = (now - m_lastUpdateTime).GetSeconds();
        
        if (deltaTime > 0)
        {
            m_totalEnergy += m_powerConsumption[m_state] * deltaTime;
            m_lastUpdateTime = now;
        }
    }

    struct StateLogEntry {
        double time;
        SmallCellState fromState;
        SmallCellState toState;
    };

    SmallCellState m_state;
    mutable double m_totalEnergy;
    mutable Time m_lastUpdateTime;
    bool m_isTransitioning;
    Time m_transitionEndTime;
    uint32_t m_nodeId;
    Ptr<energy::EnergySource> m_energySource;

    double m_powerConsumption[4];
    double m_txPower[4];
    double m_transitionTime[4];

    std::vector<StateLogEntry> m_stateLog;
};

} // namespace ns3

#endif // SMALL_CELL_ENERGY_MODEL_H
